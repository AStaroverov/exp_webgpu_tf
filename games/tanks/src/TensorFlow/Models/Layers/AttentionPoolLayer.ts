import * as tf from '@tensorflow/tfjs';

export class AttentionPoolLayer extends tf.layers.Layer {
    static className = 'AttentionPool';

    private w!: tf.LayerVariable;

    build(inputShape: tf.Shape | tf.Shape[]) {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;
        const D = shape[2];

        this.w = this.addWeight('w', [1, 1, D], 'float32', tf.initializers.glorotUniform({}));
        this.built = true;
    }

    call(tokens: tf.Tensor | tf.Tensor[]) {
        const token = Array.isArray(tokens) ? tokens[0] : tokens;
        const scores = tf.sum(tf.mul(token, this.w.read()), -1);
        const attnWeights = tf.softmax(scores);
        const attnWeightsExp = tf.expandDims(attnWeights, -1);
        const weightedTokens = tf.mul(token, attnWeightsExp);
        const pooledOutput = tf.sum(weightedTokens, 1);

        return pooledOutput;
    }
}

tf.serialization.registerClass(AttentionPoolLayer);