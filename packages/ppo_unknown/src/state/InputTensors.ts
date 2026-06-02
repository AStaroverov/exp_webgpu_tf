/**
 * InputTensors — batch `InputArrays[]` → the tf.Tensor list the network consumes.
 *
 * One input head: the board, reshaped to a spatial [B, ROWS, COLS, CHANNELS]
 * tensor so the policy/value net can run conv / spatial attention over it.
 */

import * as tf from '@tensorflow/tfjs';
import { BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, BOARD_SIZE } from './board.ts';
import type { InputArrays } from './InputArrays.ts';

export function createInputTensors(batch: InputArrays[]): tf.Tensor[] {
    const B = batch.length;
    const buf = new Float32Array(B * BOARD_SIZE);
    for (let b = 0; b < B; b++) {
        buf.set(batch[b].board, b * BOARD_SIZE);
    }
    return [
        tf.tensor4d(buf, [B, BOARD_ROWS, BOARD_COLS, BOARD_CHANNELS]),
    ];
}
