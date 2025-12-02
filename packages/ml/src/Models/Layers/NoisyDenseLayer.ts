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
     * Default: true (factorized)
     */
    factorized?: boolean;
}

/**
 * Noisy Dense Layer
 *
 * Based on "Noisy Networks for Exploration" (Fortunato et al., 2017)
 * https://arxiv.org/abs/1706.10295
 *
 * This layer replaces standard dense layers with noisy equivalents,
 * where learned noise parameters are added to the weights during forward pass.
 * This provides a form of exploration that is learned during training.
 *
 * The noisy layer can be represented as:
 * y = (μ_w + σ_w ⊙ ε_w) * x + (μ_b + σ_b ⊙ ε_b)
 *
 * Where:
 * - μ_w, μ_b are the mean weight and bias parameters
 * - σ_w, σ_b are the noise scale parameters (learned)
 * - ε_w, ε_b are noise samples (from unit Gaussian)
 * - ⊙ denotes element-wise multiplication
 */
export class NoisyDenseLayer extends tf.layers.Layer {
    static readonly className = 'NoisyDenseLayer';

    private units: number;
    private activation: Activation;
    private useBias: boolean;
    private sigma: number;
    private factorized: boolean;
    private kernelInitializer: Initializer;
    private biasInitializer: Initializer;

    // Mean parameters (μ)
    private kernelMu!: tf.LayerVariable;
    private biasMu?: tf.LayerVariable;

    // Noise scale parameters (σ)
    private kernelSigma!: tf.LayerVariable;
    private biasSigma?: tf.LayerVariable;

    private inputDim!: number;

    constructor(config: NoisyDenseConfig) {
        super(config);
        this.units = config.units;
        this.activation = getActivation(config.activation ?? 'linear');
        this.useBias = config.useBias ?? true;
        this.sigma = config.sigma ?? 0.5;
        this.factorized = config.factorized ?? true;
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
     * Generate noise samples.
     * For factorized noise: ε_w = f(ε_i) ⊗ f(ε_j) where ⊗ is outer product
     * For independent noise: ε_w is sampled independently for each weight
     */
    private sampleNoise(): { kernelNoise: tf.Tensor; biasNoise?: tf.Tensor } {
        return tf.tidy(() => {
            if (this.factorized) {
                // Factorized Gaussian noise (more efficient)
                // Sample noise for input dimension and output dimension separately
                const epsilonInput = tf.randomNormal([this.inputDim, 1]);
                const epsilonOutput = tf.randomNormal([1, this.units]);

                // Apply scaling function f(x) = sign(x) * sqrt(|x|)
                const scaledInput = this.scaleNoise(epsilonInput);
                const scaledOutput = this.scaleNoise(epsilonOutput);

                // Outer product: ε_w = f(ε_i) * f(ε_j)^T
                const kernelNoise = tf.matMul(scaledInput, scaledOutput);

                let biasNoise: tf.Tensor | undefined;
                if (this.useBias) {
                    // Bias noise uses only the output noise
                    biasNoise = scaledOutput.squeeze([0]);
                }

                return { kernelNoise, biasNoise };
            } else {
                // Independent Gaussian noise (more expressive but less efficient)
                const kernelNoise = tf.randomNormal([this.inputDim, this.units]);

                let biasNoise: tf.Tensor | undefined;
                if (this.useBias) {
                    biasNoise = tf.randomNormal([this.units]);
                }

                return { kernelNoise, biasNoise };
            }
        });
    }

    call(inputs: tf.Tensor | tf.Tensor[], kwargs?: { training?: boolean }): tf.Tensor {
        return tf.tidy(() => {
            const input = Array.isArray(inputs) ? inputs[0] : inputs;
            const training = kwargs?.training ?? true;

            // Read mean parameters
            const kernelMu = this.kernelMu.read();

            let kernel: tf.Tensor;
            let bias: tf.Tensor | undefined;

            if (training) {
                // During training: add noise
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

            // Compute output: y = x * W + b
            let output = tf.matMul(input, kernel);

            if (bias) {
                output = output.add(bias);
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
            kernelInitializer: serializeInitializer(this.kernelInitializer),
            biasInitializer: serializeInitializer(this.biasInitializer),
        };
    }

    /**
     * Reset noise by generating new noise samples.
     * Useful when you want to refresh the exploration noise.
     */
    resetNoise(): void {
        // Noise is sampled fresh on each forward pass,
        // so this method is provided for API compatibility
        // but doesn't need to do anything special
    }

    /**
     * Get the current noise scale (average of sigma parameters).
     * Useful for monitoring exploration during training.
     */
    getNoiseScale(): number {
        return tf.tidy(() => {
            const kernelSigmaValues = this.kernelSigma.read().abs().mean().arraySync() as number;
            if (this.useBias && this.biasSigma) {
                const biasSigmaValues = this.biasSigma.read().abs().mean().arraySync() as number;
                return (kernelSigmaValues + biasSigmaValues) / 2;
            }
            return kernelSigmaValues;
        });
    }
}

tf.serialization.registerClass(NoisyDenseLayer);
