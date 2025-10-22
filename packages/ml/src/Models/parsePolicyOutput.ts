import * as tf from '@tensorflow/tfjs';

export type PolicyOutput = {
    mean: tf.Tensor2D;
    phi: tf.Tensor2D;
};

export function parsePolicyOutput(prediction: tf.Tensor | tf.Tensor[]): PolicyOutput {
    const [mean, phi] = prediction as [tf.Tensor2D, tf.Tensor2D];
    return { mean, phi };
}
