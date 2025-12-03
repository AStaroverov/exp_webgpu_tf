import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import {
    getInitializer,
    Initializer,
    InitializerIdentifier,
    serializeInitializer,
} from '@tensorflow/tfjs-layers/dist/initializers';
import { getActivation, serializeActivation } from '@tensorflow/tfjs-layers/dist/activations';
import { Activation } from '@tensorflow/tfjs-layers/dist/activations';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import * as K from '@tensorflow/tfjs-layers/dist/backend/tfjs_backend';

/**
 * Noisy Dense Layer configuration
 */
export interface NoisyDenseConfig extends LayerArgs {
    /** Number of output units */
    units: number;

    /**
     * Activation function to use.
     * Default: undefined (linear activation)
     */
    activation?: ActivationIdentifier;

    /**
     * Whether to use bias.
     * Default: true
     */
    useBias?: boolean;

    /**
     * Initial standard deviation of the noise.
     * Default: 0.5 (as suggested in the paper)
     */
    sigma?: number;

    /**
     * Kernel initializer.
     * Default: 'glorotNormal'
     */
    kernelInitializer?: InitializerIdentifier | Initializer;

    /**
     * Bias initializer.
     * Default: 'zeros'
     */
    biasInitializer?: InitializerIdentifier | Initializer;

    /**
     * Whether to use factorized noise (more efficient) or independent noise.
     * Default: false (factorized)
     */
    factorized?: boolean;

    /**
     * Configuration for colored (pink) noise.
     * Uses multi-scale Ornstein-Uhlenbeck process for temporally correlated noise.
     * Default: { K: 8, tauMin: 4, tauMax: 64, beta: 1 } (pink noise)
     */
    coloredNoiseConfig?: ColoredNoiseConfig;
}

/**
 * Noisy Dense Layer with Pink (Colored) Noise
 *
 * Based on "Noisy Networks for Exploration" (Fortunato et al., 2017)
 * https://arxiv.org/abs/1706.10295
 *
 * Extended with temporally correlated (pink) noise using multi-scale
 * Ornstein-Uhlenbeck process for smoother exploration patterns.
 *
 * The noisy layer can be represented as:
 * y = (μ_w + σ_w ⊙ ε_w) * x + (μ_b + σ_b ⊙ ε_b)
 *
 * Where:
 * - μ_w, μ_b are the mean weight and bias parameters
 * - σ_w, σ_b are the noise scale parameters (learned)
 * - ε_w, ε_b are pink noise samples (temporally correlated)
 * - ⊙ denotes element-wise multiplication
 */
export class NoisyDenseLayer extends tf.layers.Layer {
    static readonly className = 'NoisyDenseLayer';

    private units: number;
    private activation: Activation;
    private useBias: boolean;
    private sigma: number;
    private factorized: boolean;
    private coloredNoiseConfig: ColoredNoiseConfig;
    private kernelInitializer: Initializer;
    private biasInitializer: Initializer;

    // Mean parameters (μ)
    private kernelMu!: tf.LayerVariable;
    private biasMu?: tf.LayerVariable;

    // Noise scale parameters (σ)
    private kernelSigma!: tf.LayerVariable;
    private biasSigma?: tf.LayerVariable;

    private inputDim!: number;

    // Colored noise state for pink noise generation
    private coloredNoiseState!: ColoredNoiseState;

    constructor(config: NoisyDenseConfig) {
        super(config);
        this.units = config.units;
        this.activation = getActivation(config.activation ?? 'linear');
        this.useBias = config.useBias ?? true;
        this.sigma = config.sigma ?? 0.5;
        this.factorized = config.factorized ?? false;
        this.coloredNoiseConfig = config.coloredNoiseConfig ?? {};
        this.kernelInitializer = config.kernelInitializer instanceof Initializer
            ? config.kernelInitializer
            : getInitializer(config.kernelInitializer ?? 'glorotNormal');
        this.biasInitializer = config.biasInitializer instanceof Initializer
            ? config.biasInitializer
            : getInitializer(config.biasInitializer ?? 'zeros');
    }

    build(inputShape: tf.Shape | tf.Shape[]): void {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;

        this.inputDim = shape[shape.length - 1]!;

        // Initialize sigma based on factorization mode
        // For factorized noise: sigma_0 / sqrt(inputDim)
        // For independent noise: sigma_0 / sqrt(inputDim)
        const sigmaInit = this.sigma / Math.sqrt(this.inputDim);

        // Mean weight parameters (μ_w)
        this.kernelMu = this.addWeight(
            'kernelMu',
            [this.inputDim, this.units],
            'float32',
            this.kernelInitializer,
            undefined,
            true,
        );

        // Noise scale parameters for weights (σ_w)
        this.kernelSigma = this.addWeight(
            'kernelSigma',
            [this.inputDim, this.units],
            'float32',
            tf.initializers.constant({ value: sigmaInit }),
            undefined,
            true,
        );

        if (this.useBias) {
            // Mean bias parameters (μ_b)
            this.biasMu = this.addWeight(
                'biasMu',
                [this.units],
                'float32',
                this.biasInitializer,
                undefined,
                true,
            );

            // Noise scale parameters for bias (σ_b)
            this.biasSigma = this.addWeight(
                'biasSigma',
                [this.units],
                'float32',
                tf.initializers.constant({ value: sigmaInit }),
                undefined,
                true,
            );
        }

        // Initialize colored noise state for pink noise generation
        this.coloredNoiseState = new ColoredNoiseState(
            this.inputDim,
            this.units,
            this.coloredNoiseConfig
        );

        super.build(inputShape);
    }

    /**
     * Scale noise for factorized Gaussian noise.
     * f(x) = sign(x) * sqrt(|x|)
     */
    private scaleNoise(x: tf.Tensor): tf.Tensor {
        return tf.sign(x).mul(tf.sqrt(tf.abs(x)));
    }

    /**
     * Generate pink noise samples using multi-scale OU process.
     * For factorized noise: ε_w = f(ε_i) ⊗ f(ε_j) where ⊗ is outer product
     * For independent noise: ε_w is computed from outer product directly
     */
    private sampleNoise(): { kernelNoise: tf.Tensor; biasNoise?: tf.Tensor } {
        if (!this.coloredNoiseState) {
            throw new Error('ColoredNoiseState not initialized');
        }

        return tf.tidy(() => {
            const { inputNoise, outputNoise } = this.coloredNoiseState.sample();

            if (this.factorized) {
                // Factorized colored noise: apply scaling function to correlated vectors
                const scaledInput = this.scaleNoise(inputNoise.reshape([this.inputDim, 1]));
                const scaledOutput = this.scaleNoise(outputNoise.reshape([1, this.units]));

                // Outer product: ε_w = f(ε_i) * f(ε_j)^T
                const kernelNoise = tf.matMul(scaledInput, scaledOutput);

                let biasNoise: tf.Tensor | undefined;
                if (this.useBias) {
                    biasNoise = this.scaleNoise(outputNoise);
                }

                // Dispose raw noise tensors
                inputNoise.dispose();
                outputNoise.dispose();

                return { kernelNoise, biasNoise };
            } else {
                // Non-factorized: use outer product directly
                const kernelNoise = tf.outerProduct(inputNoise, outputNoise);

                let biasNoise: tf.Tensor | undefined;
                if (this.useBias) {
                    biasNoise = outputNoise;
                } else {
                    outputNoise.dispose();
                }

                inputNoise.dispose();

                return { kernelNoise, biasNoise };
            }
        });
    }

    call(inputs: tf.Tensor | tf.Tensor[], kwargs?: { noise?: boolean }): tf.Tensor {
        const enable = kwargs?.noise ?? false;

        return tf.tidy(() => {
            const input = Array.isArray(inputs) ? inputs[0] : inputs;

            // Read mean parameters
            const kernelMu = this.kernelMu.read();

            let kernel: tf.Tensor;
            let bias: tf.Tensor | null = null;

            if (enable) {
                const { kernelNoise, biasNoise } = this.sampleNoise();

                // w = μ_w + σ_w ⊙ ε_w
                kernel = kernelMu.add(this.kernelSigma.read().mul(kernelNoise));

                if (this.useBias && this.biasMu && this.biasSigma && biasNoise) {
                    // b = μ_b + σ_b ⊙ ε_b
                    bias = this.biasMu.read().add(this.biasSigma.read().mul(biasNoise));
                }
            } else {
                // During inference: use only mean parameters (no noise)
                kernel = kernelMu;

                if (this.useBias && this.biasMu) {
                    bias = this.biasMu.read();
                }
            }

            // Compute output using K.dot (handles gradients correctly for any rank)
            // y = x · W + b
            let output = K.dot(input, kernel);

            if (bias !== null) {
                output = K.biasAdd(output, bias);
            }

            // Apply activation
            return this.activation.apply(output);
        });
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;

        // Replace last dimension with units
        return [...shape.slice(0, -1), this.units];
    }

    getConfig(): tf.serialization.ConfigDict {
        const config = super.getConfig();
        return {
            ...config,
            units: this.units,
            activation: serializeActivation(this.activation),
            useBias: this.useBias,
            sigma: this.sigma,
            factorized: this.factorized,
            coloredNoiseConfig: this.coloredNoiseConfig as unknown as tf.serialization.ConfigDict,
            kernelInitializer: serializeInitializer(this.kernelInitializer),
            biasInitializer: serializeInitializer(this.biasInitializer),
        };
    }

    override dispose() {
        if (this.coloredNoiseState) {
            this.coloredNoiseState.dispose();
        }
        return super.dispose();
    }
}

tf.serialization.registerClass(NoisyDenseLayer);


/**
 * Colored noise configuration for multi-scale OU process
 */
export type ColoredNoiseConfig = {
    /** Number of OU filters (scales). Default: 8 */
    K?: number;
    /** Minimum timescale in steps. Default: 4 */
    tauMin?: number;
    /** Maximum timescale in steps. Default: 64 */
    tauMax?: number;
    /** Spectral exponent (1 = pink, 2 = brown/red). Default: 1 (pink noise) */
    beta?: number;
}

/**
 * Internal state for colored noise generation using multi-scale OU process.
 * Generates temporally correlated (pink) noise for smoother exploration.
 */
class ColoredNoiseState {
    private K: number;
    private a: tf.Tensor2D;  // AR(1) coefficients
    private b: tf.Tensor2D;  // Noise scale for unit variance
    private w: tf.Tensor2D;  // Spectral weights
    private inputState: tf.Tensor2D;
    private outputState: tf.Tensor2D;
    private disposed = false;

    constructor(
        private inputDim: number,
        private outputDim: number,
        config: ColoredNoiseConfig = {}
    ) {
        const {
            K = 8,
            tauMin = 4,
            tauMax = 64,
            beta = 0.7,  // 1 = pink, 2 = brown/red
        } = config;

        this.K = K;

        const { a, b, w } = tf.tidy(() => {
            const logMin = Math.log(tauMin);
            const logMax = Math.log(tauMax);
            const logs = tf.linspace(logMin, logMax, K);
            const taus = tf.exp(logs);

            // AR(1) coefficients: a = exp(-1/tau)
            const aVec = tf.exp(tf.neg(tf.div(tf.onesLike(taus), taus)));
            // Noise scale to maintain unit variance: b = sqrt(1 - a^2)
            const bVec = tf.sqrt(tf.maximum(tf.sub(1, tf.square(aVec)), 1e-8));

            // Spectral weighting for colored noise (beta=1 -> pink, beta=2 -> brown)
            const tilt = (beta - 1) / 2;
            const wRaw = tf.pow(taus, tf.scalar(-tilt));
            const wVec = tf.div(wRaw, tf.sqrt(tf.sum(tf.square(wRaw))));

            return {
                a: aVec.reshape([K, 1]) as tf.Tensor2D,
                b: bVec.reshape([K, 1]) as tf.Tensor2D,
                w: wVec.reshape([K, 1]) as tf.Tensor2D,
            };
        });

        this.a = a;
        this.b = b;
        this.w = w;

        this.inputState = tf.randomNormal([K, inputDim]) as tf.Tensor2D;
        this.outputState = tf.randomNormal([K, outputDim]) as tf.Tensor2D;
    }

    sample(): { inputNoise: tf.Tensor1D; outputNoise: tf.Tensor1D } {
        if (this.disposed) {
            throw new Error('ColoredNoiseState: already disposed');
        }

        const { K, a, b, w, inputDim, outputDim } = this;

        // Update input state with AR(1) process
        const zInput = tf.randomNormal([K, inputDim]);
        const newInputState = a.mul(this.inputState).add(b.mul(zInput));
        const inputNoise = newInputState.mul(w).sum(0) as tf.Tensor1D;

        // Update output state with AR(1) process
        const zOutput = tf.randomNormal([K, outputDim]);
        const newOutputState = a.mul(this.outputState).add(b.mul(zOutput));
        const outputNoise = newOutputState.mul(w).sum(0) as tf.Tensor1D;


        // Update states outside tidy
        this.inputState.dispose();
        this.outputState.dispose();
        this.inputState = tf.keep(newInputState) as tf.Tensor2D;
        this.outputState = tf.keep(newOutputState) as tf.Tensor2D;

        return {
            inputNoise: inputNoise,
            outputNoise: outputNoise,
        };
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;

        this.a.dispose();
        this.b.dispose();
        this.w.dispose();
        this.inputState.dispose();
        this.outputState.dispose();
    }
}
