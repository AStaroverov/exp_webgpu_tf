import * as tf from '@tensorflow/tfjs';
import {LayerArgs} from '@tensorflow/tfjs-layers/dist/engine/topology';

export interface RMSNormConfig extends LayerArgs {
    /**
     * The axis or axes along which to compute RMS normalization.
     * Default: -1 (last axis)
     */
    axis?: number | number[];

    /**
     * Small constant added to the variance for numerical stability.
     * Default: 1e-6
     */
    epsilon?: number;

    /**
     * If true, adds a learnable scale parameter (gain).
     * Default: true
     */
    scale?: boolean;
}

/**
 * Root Mean Square Layer Normalization (RMSNorm)
 *
 * Based on "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
 * https://arxiv.org/pdf/1910.07467
 *
 * RMSNorm normalizes the inputs using the root mean square (RMS) statistic:
 *
 * RMS(x) = sqrt(mean(x^2) + epsilon)
 * output = (x / RMS(x)) * scale
 *
 * This is simpler and more efficient than LayerNorm as it doesn't center the data
 * (no mean subtraction) and only uses RMS for normalization.
 */
export class RMSNormLayer extends tf.layers.Layer {
    static readonly className = 'RMSNormLayer';

    private axis: number[];
    private epsilon: number;
    private scaleEnabled: boolean;
    private scale?: tf.LayerVariable;

    constructor(config: RMSNormConfig = {}) {
        super(config);

        // Normalize axis parameter to always be an array
        if (config.axis === undefined) {
            this.axis = [-1];
        } else if (typeof config.axis === 'number') {
            this.axis = [config.axis];
        } else {
            this.axis = config.axis;
        }

        this.epsilon = config.epsilon ?? 1e-6;
        this.scaleEnabled = config.scale ?? true;
    }

    build(inputShape: tf.Shape | tf.Shape[]): void {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;

        if (this.scaleEnabled) {
            // Determine the shape of the scale parameter
            // It should match the dimensions being normalized
            const paramShape: (null | number)[] = [];
            for (let i = 0; i < shape.length; i++) {
                if (this.axis.includes(i) || this.axis.includes(i - shape.length)) {
                    paramShape.push(shape[i]);
                } else {
                    paramShape.push(1);
                }
            }

            // Remove leading 1s and batch dimension
            const scaleShape = paramShape.slice(1);

            this.scale = this.addWeight(
                'scale',
                scaleShape,
                'float32',
                tf.initializers.ones(),
            );
        }

        super.build(inputShape);
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor | tf.Tensor[] {
        return tf.tidy(() => {
            const x = Array.isArray(inputs) ? inputs[0] : inputs;

            // Compute RMS: sqrt(mean(x^2) + epsilon)
            const squared = tf.square(x);
            const meanSquared = tf.mean(squared, this.axis, true);
            const rms = tf.sqrt(tf.add(meanSquared, this.epsilon));

            // Normalize: x / RMS(x)
            let normalized = tf.div(x, rms);

            // Apply learnable scale if enabled
            if (this.scaleEnabled && this.scale) {
                const scaleValue = this.scale.read();
                normalized = tf.mul(normalized, scaleValue);
            }

            return normalized;
        });
    }

    getConfig() {
        const baseConfig = super.getConfig();
        return {
            ...baseConfig,
            axis: this.axis.length === 1 ? this.axis[0] : this.axis,
            epsilon: this.epsilon,
            scale: this.scaleEnabled,
        };
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return inputShape;
    }
}

tf.serialization.registerClass(RMSNormLayer);

