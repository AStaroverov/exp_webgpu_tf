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

/**
 * Sample action with reparameterization trick and tanh squashing for SAC
 * Returns squashed action in [-1, 1] and corrected log probability
 */
export function sampleActionWithTanhSquashing(
    mean: tf.Tensor,
    logStd: tf.Tensor,
    epsilon: tf.Tensor, // Pre-sampled noise from N(0,1)
): { action: tf.Tensor; logProb: tf.Tensor } {
    return tf.tidy(() => {
        const std = tf.exp(logStd);

        // Reparameterization: u = mean + std * epsilon
        const unsquashedAction = tf.add(mean, tf.mul(std, epsilon));

        // Compute log prob before squashing
        const logProbUnsquashed = computeLogProb(unsquashedAction, mean, std);

        // Apply tanh squashing: action = tanh(u)
        const action = tf.tanh(unsquashedAction);

        // Correct log prob for tanh transformation
        // log π(a|s) = log π(u|s) - Σ log(1 - tanh²(u))
        // log(1 - tanh²(u)) = log(sech²(u)) = -2 * (u + log(2) - log(1 + exp(2*u)))
        const tanhSquared = tf.square(action);
        const logDet = tf.sum(
            tf.log(tf.sub(1.0, tanhSquared).add(1e-6)), // Add small epsilon for numerical stability
            -1
        );
        const logProb = tf.sub(logProbUnsquashed, logDet);

        return { action, logProb };
    });
}
