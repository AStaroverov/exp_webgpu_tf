import * as tf from '@tensorflow/tfjs';

export class AttentionPoolLayer extends tf.layers.Layer {
    static className = 'AttentionPool';

    private w!: tf.LayerVariable;
    private scale!: number;

    build(inputShape: tf.Shape | tf.Shape[]) {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;
        const D = shape[2]!;

        this.w = this.addWeight(
            'q',
            [D],
            'float32',
            tf.initializers.randomNormal({ mean: 0, stddev: 1 / Math.sqrt(D) }),
        );
        this.scale = Math.sqrt(D);

        this.built = true;
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]) {
        const shape = ((inputShape[0] === null || typeof inputShape[0] === 'number')
            ? inputShape
            : inputShape[0]) as tf.Shape;
        return [shape[0], shape[2]];
    }

    call(tokens: tf.Tensor | tf.Tensor[]) {
        const token = Array.isArray(tokens) ? tokens[0] : tokens;

        const [batch, time, features] = token.shape;
        const scores = tf.matMul(
            token.reshape([-1, features]),
            this.w.read().expandDims(-1)
        ).reshape([batch, time]).div(this.scale);
        const weights = tf.softmax(scores, -1).expandDims(-1);
        const weightedTokens = tf.mul(token, weights);
        const pooledToken = tf.sum(weightedTokens, 1);

        return pooledToken;
    }
}

tf.serialization.registerClass(AttentionPoolLayer);