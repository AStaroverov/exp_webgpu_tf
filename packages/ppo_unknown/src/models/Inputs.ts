/**
 * Network inputs for ppo_unknown — the single egocentric board map plus its
 * content mask. The returned object's value order MUST match the tensor order
 * produced by `state/InputTensors.ts`:
 *   board [ROWS, COLS, CH] → boardMask [CELLS].
 */

import * as tf from "@tensorflow/tfjs";
import { BOARD_CELLS, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS } from "./dims.ts";

export function createInputs(name: string) {
  const boardInput = tf.input({
    name: name + "_boardInput",
    shape: [BOARD_ROWS, BOARD_COLS, BOARD_CHANNELS],
  });

  // Attention content mask, 1 per cell with any non-zero channel (see
  // InputTensors.ts). Prepared outside the graph: a mask SymbolicTensor
  // consumed by several attention layers gets disposed after the first use.
  const contentMaskInput = tf.input({
    name: name + "_contentMaskInput",
    shape: [BOARD_CELLS],
  });

  return { boardInput, contentMaskInput };
}
