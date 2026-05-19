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
import { random } from '../../../../../lib/random';

/**
 * Type for noise predicate function.
 * Takes layer name and returns whether noise should be enabled for that layer.
 */
export type NoisePredicate = (layerName: string) => boolean;

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
     * Configuration for colored (pink) noise.
     * Uses multi-scale Ornstein-Uhlenbeck process for temporally correlated noise.
     * Default: { K: 8, tauMin: 4, tauMax: 64, beta: 1 } (pink noise)
     */
    coloredNoiseConfig?: ColoredNoiseConfig;

    /**
     * How to parameterize sigma to preserve pink noise correlation structure.
     * - 'full': Full [inputDim, outputDim] sigma matrix (original, destroys correlation)
     * - 'scalar': Single scalar sigma for entire layer (best correlation preservation)
     * - 'per-output': One sigma per output unit [outputDim] (preserves input correlation)
     * - 'per-input': One sigma per input unit [inputDim] (preserves output correlation)
     * Default: 'per-output'
     */
    sigmaParameterization?: 'full' | 'scalar' | 'per-output' | 'per-input';
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
    /**
     * Boost exploration by resetting noise parameters in all NoisyDenseLayer layers.
     * This helps maintain exploration when the agent gets stuck in local optima.
     */
    public static boostExploration(model: tf.LayersModel, chance: number = 0.005): void {
        let layerCount = 0;
        model.layers.forEach(layer => {
            if (layer instanceof NoisyDenseLayer && random() < chance) {
                layer.resetNoise();
                layerCount++;
            }
        });
        console.info(`[NoisyDenseLayer] Boosted ${layerCount}`);
    }

    static readonly className = 'NoisyDenseLayer';

    private units: number;
    private activation: Activation;
    private useBias: boolean;
    private sigma: number;
    private coloredNoiseConfig: ColoredNoiseConfig;
    private sigmaParameterization: 'full' | 'scalar' | 'per-output' | 'per-input';
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
        this.sigma = config.sigma ?? 0.25;
        this.coloredNoiseConfig = config.coloredNoiseConfig ?? {};
        this.sigmaParameterization = config.sigmaParameterization ?? 'full';
        this.kernelInitializer = config.kernelInitializer instanceof Initializer
            ? config.kernelInitializer
            : getInitializer(config.kernelInitializer ?? 'glorotNormal');
        this.biasInitializer = config.biasInitializer instanceof Initializer
            ? config.biasInitializer
            : getInitializer(config.biasInitializer ?? 'zeros');
    }

    /**
     * Resets or scales the noise parameters (sigma) to boost exploration.
     * 
     * @param scaleFactor Optional multiplier for the current sigma values. 
     *                    If not provided, resets to initial sigma configuration.
     */
    public resetNoise(scaleFactor?: number): void {
        tf.tidy(() => {
            if (scaleFactor !== undefined) {
                // Multiply current sigma by a factor (e.g., 1.5 to increase noise by 50%)
                const newKernelSigma = this.kernelSigma.read().mul(scaleFactor);
                this.kernelSigma.write(newKernelSigma);

                if (this.useBias && this.biasSigma) {
                    const newBiasSigma = this.biasSigma.read().mul(scaleFactor);
                    this.biasSigma.write(newBiasSigma);
                }
            } else {
                const sigmaInit = this.sigma / Math.sqrt(this.inputDim);
                // Hard reset to initial values (aggressive exploration restart)
                const kernelSigmaShape = this.getSigmaShape('kernel');
                const newKernelSigma = tf.fill(kernelSigmaShape, sigmaInit);
                this.kernelSigma.write(newKernelSigma);

                if (this.useBias && this.biasSigma) {
                    const newBiasSigma = tf.fill([this.units], sigmaInit);
                    this.biasSigma.write(newBiasSigma);
                }
            }
        });
    }

    build(inputShape: tf.Shape | tf.Shape[]): void {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;

        this.inputDim = shape[shape.length - 1]!;

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
        // Shape depends on sigmaParameterization to preserve pink noise correlation
        const kernelSigmaShape = this.getSigmaShape('kernel');
        this.kernelSigma = this.addWeight(
            'kernelSigma',
            kernelSigmaShape,
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

            // Noise scale parameters for bias (σ_b) - always per-output
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
     * Get sigma shape based on parameterization strategy.
     */
    private getSigmaShape(type: 'kernel' | 'bias'): number[] {
        if (type === 'bias') {
            return [this.units];
        }

        switch (this.sigmaParameterization) {
            case 'scalar':
                return [1];
            case 'per-output':
                return [this.units];
            case 'per-input':
                return [this.inputDim];
            case 'full':
            default:
                return [this.inputDim, this.units];
        }
    }

    /**
     * Apply sigma to noise tensor, preserving correlation structure where possible.
     */
    private applySigmaToNoise(noise: tf.Tensor, sigma: tf.Tensor, type: 'kernel' | 'bias'): tf.Tensor {
        if (type === 'bias') {
            return noise.mul(sigma);
        }

        switch (this.sigmaParameterization) {
            case 'scalar':
                // Single scalar scales entire noise matrix uniformly - best for correlation
                return noise.mul(sigma);
            case 'per-output':
                // [units] sigma broadcasts to [inputDim, units] - preserves input correlation
                return noise.mul(sigma.reshape([1, this.units]));
            case 'per-input':
                // [inputDim] sigma broadcasts to [inputDim, units] - preserves output correlation
                return noise.mul(sigma.reshape([this.inputDim, 1]));
            case 'full':
            default:
                // Full element-wise multiplication - destroys correlation (original behavior)
                return noise.mul(sigma);
        }
    }

    /**
     * Generate pink noise samples using multi-scale OU process.
     */
    private sampleNoise(): { kernelNoise: tf.Tensor; biasNoise?: tf.Tensor } {
        if (!this.coloredNoiseState) {
            throw new Error('ColoredNoiseState not initialized');
        }

        return tf.tidy(() => {
            const { kernelNoise, biasNoise } = this.coloredNoiseState.sample();
            return { kernelNoise, biasNoise };
        });
    }

    call(inputs: tf.Tensor | tf.Tensor[], kwargs?: { noise?: boolean | NoisePredicate }): tf.Tensor {
        // Determine if noise should be enabled:
        // - undefined: no noise
        // - boolean: direct enable/disable
        // - function (predicate): call with layer name
        const noiseArg = kwargs?.noise;
        const enable = noiseArg === undefined
            ? false
            : typeof noiseArg === 'boolean'
                ? noiseArg
                : noiseArg(this.name);

        return tf.tidy(() => {
            const input = Array.isArray(inputs) ? inputs[0] : inputs;

            // Read mean parameters
            const kernelMu = this.kernelMu.read();

            let kernel: tf.Tensor;
            let bias: tf.Tensor | null = null;

            if (enable) {
                const { kernelNoise, biasNoise } = this.sampleNoise();

                // w = μ_w + σ_w ⊙ ε_w (with sigma applied preserving correlation)
                kernel = kernelMu.add(this.applySigmaToNoise(kernelNoise, this.kernelSigma.read(), 'kernel'));

                if (this.useBias && this.biasMu && this.biasSigma && biasNoise) {
                    // b = μ_b + σ_b ⊙ ε_b
                    bias = this.biasMu.read().add(this.applySigmaToNoise(biasNoise, this.biasSigma.read(), 'bias'));
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
            coloredNoiseConfig: this.coloredNoiseConfig as unknown as tf.serialization.ConfigDict,
            sigmaParameterization: this.sigmaParameterization,
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
 * Colored noise configuration
 */
export type ColoredNoiseConfig = {
    /** 
     * Smoothing factor (0 to 1). 
     * Higher = smoother noise (more temporal correlation).
     * 0 = white noise, 0.9+ = very smooth
     * Default: 0.9
     */
    smoothing?: number;
}

/**
 * Simple colored noise state using exponential moving average (EMA).
 * 
 * Formula: noise_t = α * noise_{t-1} + √(1-α²) * white_noise_t
 * 
 * This creates temporally correlated noise where α controls smoothness.
 * The √(1-α²) factor ensures unit variance is preserved.
 */
class ColoredNoiseState {
    private alpha: number;
    private sigma: number;
    private kernelState: tf.Tensor2D;
    private biasState: tf.Tensor1D;
    private disposed = false;

    constructor(
        private inputDim: number,
        private outputDim: number,
        config: ColoredNoiseConfig = {}
    ) {
        // α controls how much of previous noise is retained
        this.alpha = config.smoothing ?? 0.75;
        // σ ensures unit variance: Var = σ²/(1-α²) = 1, so σ = √(1-α²)
        this.sigma = Math.sqrt(1 - this.alpha * this.alpha);

        // Initialize with random noise
        this.kernelState = tf.randomNormal([inputDim, outputDim]) as tf.Tensor2D;
        this.biasState = tf.randomNormal([outputDim]) as tf.Tensor1D;
    }

    sample(): { kernelNoise: tf.Tensor2D; biasNoise: tf.Tensor1D } {
        if (this.disposed) {
            throw new Error('ColoredNoiseState: already disposed');
        }

        const result = tf.tidy(() => {
            // EMA update: new_state = α * old_state + σ * white_noise
            const whiteKernel = tf.randomNormal([this.inputDim, this.outputDim]);
            const newKernelState = this.kernelState
                .mul(this.alpha)
                .add(whiteKernel.mul(this.sigma)) as tf.Tensor2D;

            const whiteBias = tf.randomNormal([this.outputDim]);
            const newBiasState = this.biasState
                .mul(this.alpha)
                .add(whiteBias.mul(this.sigma)) as tf.Tensor1D;

            return {
                kernelNoise: tf.keep(newKernelState),
                biasNoise: tf.keep(newBiasState),
            };
        });

        // Update states
        this.kernelState.dispose();
        this.biasState.dispose();
        this.kernelState = result.kernelNoise;
        this.biasState = result.biasNoise;

        return { kernelNoise: result.kernelNoise, biasNoise: result.biasNoise };
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;

        this.kernelState.dispose();
        this.biasState.dispose();
    }
}
