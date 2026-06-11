/**
 * HexRingGatherLayer — tokens [B, CELLS, D] → gathered tokens [B, N, D]: the
 * tokens at the N constant cell `indexes` given to the constructor, in that
 * order. A plain fixed-index `tf.gather` along the token axis.
 *
 * Used in v4 with `ACTION_CELL_INDEXES` (state/hexNeighbors.ts): the board is
 * EGOCENTRIC (self is ALWAYS the window center, see board.ts), so the 37
 * action cells — self + the fire-target rings — sit at fixed window indexes;
 * no self-plane lookup is needed.
 */

import * as tf from "@tensorflow/tfjs";
import { Layer } from "@tensorflow/tfjs-layers/dist/engine/topology";
import type { LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";

export class HexRingGatherLayer extends Layer {
  static readonly className = "HexRingGather";

  private indexes: number[];
  private indexesTensor?: tf.Tensor1D;

  constructor(config: LayerArgs & { indexes: readonly number[] }) {
    super(config);
    this.indexes = [...config.indexes];
  }

  build(inputShape: tf.Shape | tf.Shape[]) {
    this.indexesTensor = tf.tensor1d(this.indexes, "int32");
    super.build(inputShape);
  }

  computeOutputShape(inputShape: tf.Shape | tf.Shape[]) {
    const shape = (Array.isArray(inputShape[0]) ? inputShape[0] : inputShape) as tf.Shape;
    return [shape[0], this.indexes.length, shape[2]];
  }

  call(inputs: tf.Tensor | tf.Tensor[]) {
    const tokens = Array.isArray(inputs) ? inputs[0] : inputs; // [B, CELLS, D]
    return tf.gather(tokens, this.indexesTensor!, 1); // [B, N, D]
  }

  getConfig() {
    const config = super.getConfig();
    return { ...config, indexes: this.indexes };
  }

  dispose() {
    this.indexesTensor?.dispose();
    return super.dispose();
  }
}

tf.serialization.registerClass(HexRingGatherLayer);
