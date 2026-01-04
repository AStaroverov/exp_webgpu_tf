import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

export class MaskLikeLayer extends tf.layers.Layer {
    static readonly className = 'MaskLike';

    constructor(config?: LayerArgs) {
        super(config || {});
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;

        // Return shape [batch, seqLen]
        return [shape[0], shape[1]];
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
        return tf.tidy(() => {
            const input = Array.isArray(inputs) ? inputs[0] : inputs;
            const [batch, seqLen] = input.shape;

            // Create a mask filled with ones
            return tf.ones([batch, seqLen]);
        });
    }

    getConfig() {
        const config = super.getConfig();
        return config;
    }
}

tf.serialization.registerClass(MaskLikeLayer);
