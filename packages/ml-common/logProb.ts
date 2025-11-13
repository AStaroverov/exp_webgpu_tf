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

export function computeLogProbTanh(
    actions: tf.Tensor, // [batch, actDim], в tanh-пространстве
    mean: tf.Tensor,    // [batch, actDim], μ
    std: tf.Tensor      // [batch, actDim], σ (> 0)
) {
    const eps = 1e-4;

    // Transform actions from tanh space to Gaussian space
    const clipped = tf.clipByValue(actions, -1 + eps, 1 - eps);
    const u = tf.atanh(clipped);

    // Standardized distance from mean
    const z = u.sub(mean).div(std); // (u - μ) / σ

    // Log-probability components for Gaussian
    const zSquared = z.square();                     // z²
    const logTwoPi = tf.scalar(Math.log(2 * Math.PI));
    const logVar = std.square().log();              // log(σ²) = 2 log σ

    // Log N(u | μ, σ²) per-dimension
    const logGaussian = tf.scalar(-0.5).mul(
        zSquared.add(logTwoPi).add(logVar)
    ); // shape: [batch, actDim]

    // Jacobian correction: -log(1 - a²)
    const logJacobian = tf
        .scalar(1)
        .sub(clipped.square())
        .add(eps)
        .log(); // log(1 - a²)

    // Sum over action dimensions → [batch]
    const logProbGaussian = logGaussian.sum(1);
    const logDetJacobian = logJacobian.sum(1);

    return logProbGaussian.sub(logDetJacobian); // [batch]
}

