/**
 * HexNeighborGatherLayer — [encoded [B, CELLS, D], selfPlane [B, CELLS]] →
 * neighbor tokens [B, 6, D]: the encoded token of each hex neighbor of the
 * self cell, in POINTY_DIRECTIONS order (the same order the move/fire action
 * slices use, see consts.ts / computeActionMask).
 *
 * The gather is two matmuls against a constant neighbor matrix built once from
 * the game's hex topology (honeycomb, row-parity offsets included):
 *   planes[b, d] = selfPlane[b] × S_d   — one-hot of the d-neighbor cell
 *   tokens[b, d] = planes[b, d] × encoded[b]
 * Off-board neighbors have an all-zero row → zero token (those directions are
 * masked by computeActionMask anyway).
 */

import * as tf from "@tensorflow/tfjs";
import { NEIGHBOR_DIRS as DIRS, NEIGHBOR_INDEX } from "../../state/hexNeighbors.ts";
import { BOARD_CELLS } from "../dims.ts";

// [CELLS, DIRS*CELLS]: row c, column d*CELLS+n = 1 ⇔ cell n is the d-neighbor of c.
function buildNeighborMatrix(): Float32Array {
  const m = new Float32Array(BOARD_CELLS * DIRS * BOARD_CELLS);
  for (let c = 0; c < BOARD_CELLS; c++) {
    for (let d = 0; d < DIRS; d++) {
      const n = NEIGHBOR_INDEX[c * DIRS + d];
      if (n >= 0) m[c * DIRS * BOARD_CELLS + d * BOARD_CELLS + n] = 1;
    }
  }
  return m;
}

export class HexNeighborGatherLayer extends tf.layers.Layer {
  static readonly className = "HexNeighborGather";

  private matrix?: tf.Tensor2D;

  build(inputShape: tf.Shape | tf.Shape[]) {
    this.matrix = tf.tensor2d(buildNeighborMatrix(), [BOARD_CELLS, DIRS * BOARD_CELLS]);
    super.build(inputShape);
  }

  computeOutputShape(inputShape: tf.Shape[]) {
    const [encShape] = inputShape;
    return [encShape[0], DIRS, encShape[2]];
  }

  call(inputs: tf.Tensor | tf.Tensor[]) {
    return tf.tidy(() => {
      const [encoded, selfPlane] = inputs as tf.Tensor[]; // [B,CELLS,D], [B,CELLS]
      const planes = tf
        .matMul(selfPlane as tf.Tensor2D, this.matrix!)
        .reshape([-1, DIRS, BOARD_CELLS]); // [B,DIRS,CELLS]
      return tf.matMul(planes, encoded as tf.Tensor3D); // [B,DIRS,D]
    });
  }

  dispose() {
    this.matrix?.dispose();
    return super.dispose();
  }
}

tf.serialization.registerClass(HexNeighborGatherLayer);
