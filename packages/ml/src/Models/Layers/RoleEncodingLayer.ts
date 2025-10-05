import * as tf from '@tensorflow/tfjs';

export class RoleEmbeddingLayer extends tf.layers.Layer {
    static readonly className = 'RoleEmbeddingLayer';

    private roleVec!: tf.LayerVariable;

    build(inputShape: tf.Shape | tf.Shape[]) {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;

        const dModel = shape[shape.length - 1]!;

        this.roleVec = this.addWeight(
            'role_vec',
            [1, 1, dModel],
            'float32',
            tf.initializers.randomNormal({ stddev: 0.02 }),
        );
        this.built = true;
    }

    call(inputs: tf.Tensor | tf.Tensor[]) {
        const x = Array.isArray(inputs) ? inputs[0] : inputs; // [B,N,d]
        return tf.add(x, this.roleVec.read());
    }
}

tf.serialization.registerClass(RoleEmbeddingLayer);
