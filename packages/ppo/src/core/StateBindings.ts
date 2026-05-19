import * as tf from '@tensorflow/tfjs';

export interface StateBindings<S> {
    /** Build the per-input tf.Tensor list (one entry per network input head) from a batch of states. */
    createInputTensors(batch: S[]): tf.Tensor[];
    /** Produce a randomly-initialised state of shape S (for warmup / shape probes). */
    prepareRandomInputArrays(): S;
}
