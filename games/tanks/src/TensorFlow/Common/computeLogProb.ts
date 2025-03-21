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

export function computeLogProbTanh(actions: tf.Tensor, mean: tf.Tensor, std: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
        let logProbRaw = computeLogProb(actions, mean, std);

        // 2) Якобиан "tanh": вычитаем sum(log(1 - a^2))
        //    добавляем eps, чтобы избежать log(0)
        const eps = 1e-6;
        const logDetJacobian = tf.sum(
            tf.log(tf.scalar(1).sub(actions.tanh().square()).add(eps)),
            -1,
        );

        // logProb = гауссовская часть - \sum log(1 - a^2)
        // (т.к. log p(a) = log p(rawAct) + log|d rawAct / d a|,
        //  а log|d rawAct / d a| = - log(1 - a^2))
        const logProb = tf.sub(logProbRaw, logDetJacobian);

        return logProb;
    });
}