import * as tf from "@tensorflow/tfjs";
import { GradSaveFunc } from "@tensorflow/tfjs-core";

/**
 * Custom gatherND with gradient support for training.
 * The standard tf.gatherND doesn't have gradient registered on all backends.
 */
export function gatherND<T extends tf.Tensor>(x: T, indices: tf.Tensor): tf.Tensor {
  // Use customGrad to define forward pass and gradient together
  const customGatherND = tf.customGrad(
    (x: tf.Tensor | GradSaveFunc, save: tf.Tensor | GradSaveFunc) => {
      const input = x as tf.Tensor;
      const saveFn = save as GradSaveFunc;

      // Save indices for backward pass
      saveFn([indices]);

      // Forward pass: standard gatherND
      const result = tf.gatherND(input, indices);

      // Gradient function
      const gradFunc = (dy: tf.Tensor, saved: tf.Tensor[]): tf.Tensor => {
        const [savedIndices] = saved;
        // Scatter gradients back to original positions
        return tf.scatterND(savedIndices, dy, input.shape);
      };

      return { value: result, gradFunc };
    },
  );

  return customGatherND(x);
}
