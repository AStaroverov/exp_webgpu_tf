/**
 * InputArrays — the per-decision observation `S` for ppo_unknown.
 *
 * Unlike tanks (many physics groups), the unknown game is chess-like: a single
 * multi-plane board IS the whole observation (see board.ts / snapshotUnknownBoard).
 * `S` is therefore just a flat copy of one tank's board buffer — plain typed array,
 * cheap to clone into `AgentMemory` and to ship across the learner channel.
 */

import { BOARD_SIZE, UnknownInputBoard } from './board.ts';

export type InputArrays = {
    /** Flat board buffer, cell-major [ROWS, COLS, CHANNELS]; length BOARD_SIZE. */
    board: Float32Array;
};

/**
 * Snapshot one observer's board out of the SoA store into a standalone `S`.
 * Prereq: `snapshotUnknownBoard` must have run this tick so the SoA row is fresh.
 */
export function prepareInputArrays(eid: number): InputArrays {
    const view = UnknownInputBoard.board.getBatch(eid); // f64 view, length BOARD_SIZE
    const board = new Float32Array(BOARD_SIZE);
    board.set(view);
    return { board };
}

/** Random state for warmup / shape probes (matches the real board layout). */
export function prepareRandomInputArrays(): InputArrays {
    const board = new Float32Array(BOARD_SIZE);
    for (let i = 0; i < board.length; i++) board[i] = Math.random();
    return { board };
}
