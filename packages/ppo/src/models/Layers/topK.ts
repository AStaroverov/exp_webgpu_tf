import * as tf from "@tensorflow/tfjs";
import { GradSaveFunc } from "@tensorflow/tfjs-core";

/**
 * Custom topK with gradient support for training.
 * The standard tf.topk doesn't have gradient registered on all backends.
 *
 * Returns { values, indices } where values have proper gradient support.
 */
export function topK(x: tf.Tensor, k: number): { values: tf.Tensor; indices: tf.Tensor } {
  // First, compute topK to get indices (we need these for both forward and backward)
  const { indices } = tf.topk(x, k);

  // Use customGrad for the values computation with proper gradients
  const customTopKValues = tf.customGrad(
    (input: tf.Tensor | GradSaveFunc, save: tf.Tensor | GradSaveFunc) => {
      const x = input as tf.Tensor;
      const saveFn = save as GradSaveFunc;

      // Save input shape and indices for backward pass
      saveFn([indices]);

      // Forward pass: get top-K values using gather
      // indices shape: [batch, k]
      // x shape: [batch, n]
      const batchSize = x.shape[0]!;
      const batchIndices = tf.range(0, batchSize, 1, "int32").expandDims(1).tile([1, k]);

      // Create gather indices: [batch * k, 2]
      const gatherIndices = tf.stack([batchIndices.flatten(), indices.flatten().cast("int32")], 1);

      // Gather values
      const values = tf.gatherND(x, gatherIndices).reshape([batchSize, k]);

      // Gradient function: scatter gradients back to original positions
      const gradFunc = (dy: tf.Tensor, saved: tf.Tensor[]): tf.Tensor => {
        const [savedIndices] = saved;
        const dyFlat = dy.flatten();

        // Create scatter indices
        const batchIdx = tf.range(0, batchSize, 1, "int32").expandDims(1).tile([1, k]);
        const scatterIndices = tf.stack(
          [batchIdx.flatten(), savedIndices.flatten().cast("int32")],
          1,
        );

        // Scatter gradients back
        return tf.scatterND(scatterIndices, dyFlat, x.shape);
      };

      return { value: values, gradFunc };
    },
  );

  const values = customTopKValues(x);

  return { values, indices };
}
