import * as tf from '@tensorflow/tfjs';
import { Tensor } from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

export class AttentionMaskLayer extends tf.layers.Layer {
    constructor(config?: LayerArgs) {
        super(config);
    }

    static get className() {
        return 'AttentionMaskLayer';
    }

    call(inputs: Tensor[] | Tensor): Tensor {
        const [scores, mask] = inputs as Tensor[]; // scores: [batch, 1, slots], mask: [batch, slots]

        return tf.tidy(() => {
            const one = tf.scalar(1);
            const negInf = tf.scalar(-1e9);

            if (mask.shape.length !== 2) {
                throw new Error('mask shape must be 2D');
            }
            const maskExpanded = mask.reshape([-1, 1, mask.shape[1]]); // [batch, 1, slots]
            const inverted = tf.sub(one, maskExpanded);                // 1 - mask
            const penalty = tf.mul(inverted, negInf);                  // (1 - mask) * -1e9

            return tf.add(scores, penalty); // scores + penalty
        });
    }

    computeOutputShape(inputShape: [number[], number[]]): number[] {
        return inputShape[0]; // output shape = scores shape
    }

    getConfig() {
        return { ...super.getConfig() };
    }
}

tf.serialization.registerClass(AttentionMaskLayer);
