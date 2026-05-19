import * as tf from '@tensorflow/tfjs';
import { Variable } from '@tensorflow/tfjs';
import { mul } from '@tensorflow/tfjs-core/dist/ops/mul';
import { PatchedAdamOptimizer } from './PatchedAdamOptimizer';

export class AdamW extends PatchedAdamOptimizer {
    static className = 'AdamW';

    private weightDecay: number;

    constructor(
        learningRate = 0.001,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon?: number,
        weightDecay = 1e-6
    ) {
        super(learningRate, beta1, beta2, epsilon);
        this.weightDecay = weightDecay;
    }

    protected computeAdditionalUpdate(value: Variable, name: string): tf.Tensor | null {
        name = name.toLowerCase();

        if (name.includes('/bias') || name.endsWith('bias')) return null;
        if (name.includes('layernorm') || name.includes('layer_norm')) return null; // γ/β
        if (name.includes('batchnorm') || name.includes('batch_normalization')) return null; // γ/β
        // custom layers
        if (name.includes('rmsnormlayer')) return null;
        if (name.includes('noisydenselayer') && name.includes('sigma')) return null; // noisy layer sigmas

        // AdamW: Apply decoupled weight decay
        // weight_decay = -learning_rate * weight_decay_rate * weights
        return mul(value, -(this.learningRate * this.weightDecay));
    }

    getConfig() {
        return {
            ...super.getConfig(),
            weightDecay: this.weightDecay,
        };
    }

    static fromConfig<T extends tf.serialization.Serializable>(
        cls: tf.serialization.SerializableConstructor<T>,
        config: tf.serialization.ConfigDict
    ): T {
        return new cls(
            config['learningRate'] as number,
            config['beta1'] as number,
            config['beta2'] as number,
            config['epsilon'] as number,
            config['weightDecay'] as number
        );
    }
}

tf.serialization.registerClass(AdamW);
