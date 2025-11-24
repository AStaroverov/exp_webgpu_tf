import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

export class MaskSquashLayer extends tf.layers.Layer {
    static readonly className = 'MaskSquashLayer';

    constructor(config?: LayerArgs) {
        super(config || {});
    }

    getConfig() {
        const config = super.getConfig();
        return config;
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;

        return [shape[0], 1];
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
        const input = Array.isArray(inputs) ? inputs[0] : inputs;
        const result = input.max(1, true);
        return result;
    }
}

tf.serialization.registerClass(MaskSquashLayer);
