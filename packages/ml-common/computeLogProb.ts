import * as tf from '@tensorflow/tfjs';

export function computeLogProb1(actions: tf.Tensor, mean: tf.Tensor, std: tf.Tensor): tf.Tensor {
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


const LOG_2PI = Math.log(2.0 * Math.PI);

export function computeLogProb(
    actions: tf.Tensor, // [B, A]
    mean: tf.Tensor,    // [B, A]
    std: tf.Tensor      // [B, A]
): tf.Tensor {
    return tf.tidy(() => {
        const diff = actions.sub(mean);                     // [B, A]
        const variance = std.square();                      // σ^2
        const quad = diff.square().div(variance).sum(-1);   // [B]
        const logDet = std.log().mul(2).sum(-1);            // [B], = Σ log(σ^2) = 2*Σ log(σ)
        // -0.5 * (quad + log( (2π)^A * |Σ| ))
        return quad.add(logDet).add(LOG_2PI * (actions.shape[1] ?? 1)).mul(-0.5);
    });
}