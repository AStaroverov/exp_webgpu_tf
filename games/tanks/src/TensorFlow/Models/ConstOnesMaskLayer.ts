import * as tf from '@tensorflow/tfjs';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

export class OnesMask extends tf.layers.Layer {
    private length: number;

    constructor(args: LayerArgs & { length?: number }) {
        super(args);
        this.length = args.length ?? 1;
    }

    static get className() {
        return 'OnesMask';
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
        const batch = (inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape[0]
            : inputShape[0][0];
        return [batch, this.length];
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
        const tensor = Array.isArray(inputs) ? inputs[0] as tf.Tensor : inputs as tf.Tensor;
        return tf.ones([tensor.shape[0], this.length], tensor.dtype);
    }

    getConfig() {
        return {
            ...super.getConfig(),
            length: this.length,
        };
    }
}

tf.serialization.registerClass(OnesMask);
