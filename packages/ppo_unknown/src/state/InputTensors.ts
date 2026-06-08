/**
 * InputTensors — batch `InputArrays[]` → the tf.Tensor list the network consumes.
 *
 * Tensor ORDER is the contract with the model side (`models/Inputs.ts` must keep
 * `Object.values(inputs)` in the SAME order):
 *   [0] board     [B, ROWS, COLS, CHANNELS] — egocentric multi-plane window
 *   [1] boardMask [B, CELLS]                — per-cell content mask for the board
 *
 * boardMask (prepared OUTSIDE the graph, like the board mask always was — see the
 * SymbolicTensor-dispose note in `models/Inputs.ts`): a cell is "content" if ANY
 * channel OTHER than the always-on coordinate planes (`CoordX`/`CoordY`) is
 * non-zero. The raw channel-sum mask alone would not work: `CoordX/CoordY` are
 * written for every in-view cell, so the sum is non-zero everywhere — we exclude
 * those two planes.
 */

import * as tf from '@tensorflow/tfjs';
import { BOARD_CELLS, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, BOARD_SIZE, BoardChannel } from './board.ts';
import type { InputArrays } from './InputArrays.ts';

const CONTENT_CHANNELS = Array.from({ length: BOARD_CHANNELS }, (_, ch) => ch)
    .filter((ch) => ch !== BoardChannel.CoordX && ch !== BoardChannel.CoordY);

export function createInputTensors(batch: InputArrays[]): tf.Tensor[] {
    const B = batch.length;
    const boardBuf = new Float32Array(B * BOARD_SIZE);
    const maskBuf = new Float32Array(B * BOARD_CELLS);

    for (let b = 0; b < B; b++) {
        const board = batch[b].board;
        boardBuf.set(board, b * BOARD_SIZE);
        for (let c = 0; c < BOARD_CELLS; c++) {
            const base = c * BOARD_CHANNELS;
            const hasContent = CONTENT_CHANNELS.some((ch) => board[base + ch] !== 0);
            maskBuf[b * BOARD_CELLS + c] = hasContent ? 1 : 0;
        }
    }
    return [
        tf.tensor4d(boardBuf, [B, BOARD_ROWS, BOARD_COLS, BOARD_CHANNELS]),
        tf.tensor2d(maskBuf, [B, BOARD_CELLS]),
    ];
}
