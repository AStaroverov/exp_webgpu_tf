/**
 * InputTensors — batch `InputArrays[]` → the tf.Tensor list the network consumes.
 *
 * One input head: the board, reshaped to a spatial [B, ROWS, COLS, CHANNELS]
 * tensor so the policy/value net can run conv / spatial attention over it.
 */

import * as tf from '@tensorflow/tfjs';
import { BOARD_CELLS, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, BOARD_SIZE } from './board.ts';
import type { InputArrays } from './InputArrays.ts';

export function createInputTensors(batch: InputArrays[]): tf.Tensor[] {
    const B = batch.length;
    const buf = new Float32Array(B * BOARD_SIZE);
    const maskBuf = new Float32Array(B * BOARD_CELLS);

    for (let b = 0; b < B; b++) {
        const board = batch[b].board;
        buf.set(board, b * BOARD_SIZE);
        for (let c = 0; c < BOARD_CELLS; c++) {
            const cell = board.subarray(c * BOARD_CHANNELS, (c + 1) * BOARD_CHANNELS);
            const sum = cell.reduce((acc, v) => acc + v, 0);
            maskBuf[b * BOARD_CELLS + c] = sum > 0 ? 1 : 0;
        }
    }
    return [
        tf.tensor4d(buf, [B, BOARD_ROWS, BOARD_COLS, BOARD_CHANNELS]),
        tf.tensor2d(maskBuf, [B, BOARD_CELLS]),
    ];
}
