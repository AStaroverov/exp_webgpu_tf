import * as tf from '@tensorflow/tfjs';

export function computeLogProb(actions: tf.Tensor, mean: tf.Tensor, std: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
        const logUnnormalized = tf.mul(
            -0.5,
            tf.square(
                tf.sub(
                    tf.div(actions, std),
                    tf.div(mean, std),
                ),
            ),
        );
        const logNormalization = tf.add(
            tf.scalar(0.5 * Math.log(2.0 * Math.PI)),
            tf.log(std),
        );
        return tf.sum(
            tf.sub(logUnnormalized, logNormalization),
            logUnnormalized.shape.length - 1,
        );
    });
}

export function computeLogProbTanh(actions: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) {
    const u = tf.atanh(tf.clipByValue(actions, -0.999999, 0.999999));
    const z = u.sub(mean).div(std);
    const logN = tf.scalar(-0.5).mul(z.square().add(tf.log(tf.scalar(2 * Math.PI)).add(std.square().log().mul(2))));
    const logJac = tf.log(tf.scalar(1).sub(actions.square()).add(1e-6)); // ∑ log(1 - a^2)
    return logN.sum(1).sub(logJac.sum(1)); // [batch]
}
