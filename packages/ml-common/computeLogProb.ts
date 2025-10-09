import * as tf from '@tensorflow/tfjs';

// export function computeLogProb(actions: tf.Tensor, mean: tf.Tensor, std: tf.Tensor): tf.Tensor {
//     return tf.tidy(() => {
//         const logUnnormalized = tf.mul(
//             -0.5,
//             tf.square(
//                 tf.sub(
//                     tf.div(actions, std),
//                     tf.div(mean, std),
//                 ),
//             ),
//         );
//         const logNormalization = tf.add(
//             tf.scalar(0.5 * Math.log(2.0 * Math.PI)),
//             tf.log(std),
//         );
//         return tf.sum(
//             tf.sub(logUnnormalized, logNormalization),
//             logUnnormalized.shape.length - 1,
//         );
//     });
// }


const LOG_2PI = Math.log(2.0 * Math.PI);

export function computeLogProb(
    actions: tf.Tensor, // [B, A]
    mean: tf.Tensor,    // [B, A]
    logStd: tf.Tensor   // [B, A] или [A] или [B, A]
): tf.Tensor {
    return tf.tidy(() => {
        const diff = actions.sub(mean);                     // [B, A]
        const invVar = tf.exp(logStd.mul(-2));              // 1/σ^2
        const quad = diff.square().mul(invVar).sum(-1);     // [B]
        const logDet = logStd.mul(2).sum(-1);               // [B], = Σ log(σ^2)
        // (quad + log( (2π)^A * |Σ| ) ) * -0.5
        return quad.add(logDet).add(LOG_2PI * (actions.shape[1] ?? 1)).mul(-0.5);
    });
}