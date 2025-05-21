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

export function computeLogProbTanh(actions: tf.Tensor, mean: tf.Tensor, logStd: tf.Tensor) {
    return tf.tidy(() => {
        // 1. "Unsquash": z = atanh(a)
        const EPS = 1e-6;                         // защита от ±1
        const clipped = tf.clipByValue(actions, -1 + EPS, 1 - EPS);
        const preTanh = tf.atanh(clipped);        // shape = [...]

        // 2. log N(z | μ, σ²)
        const log2pi = tf.scalar(Math.log(2 * Math.PI));
        const squareEps = logStd.mul(2).exp();       // σ²

        const logProbGaussian = preTanh.sub(mean)      // (z-μ)
            .square()
            .div(squareEps)                               // /σ²
            .add(logStd.mul(2))                      // +2 ln σ
            .add(log2pi)                             // +ln 2π
            .mul(-0.5)                               // –½ …
            .sum(-1, true);                          // sum over act-dim

        // 3. Якобиан tanh:  Σ log(1 − a²)
        const logDetJac = tf.log(tf.scalar(1).sub(clipped.square()))
            .sum(-1, true);

        // 4. Финальный log π
        return logProbGaussian.sub(logDetJac);       // shape [batch,1]
    });
}