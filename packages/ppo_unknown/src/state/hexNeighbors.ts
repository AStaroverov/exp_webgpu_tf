/**
 * Flat board-index hex neighbor table for the EGOCENTRIC window (see board.ts).
 * The window is indexed by AXIAL deltas (col = dq + R, row = dr + R), and axial
 * neighbor offsets are parity-free constants — so the table is uniform shifts,
 * -1 at the window edge. Direction order = POINTY_DIRECTIONS, the same order the
 * move/fire action slices use (see consts.ts).
 *
 * The per-direction axial deltas are derived from honeycomb itself (not
 * hardcoded) so they can never drift from the game's hex topology.
 *
 * Used graph-side (HexNeighborGatherLayer's constant gather matrix); only the
 * center cell's row is actually read there (self sits at the window center).
 */

import { Grid, rectangle } from 'honeycomb-grid';
import { HexTile, POINTY_DIRECTIONS } from '../../../unknown/src/Game/Map/HexConfig.ts';
import { BOARD_CELLS, BOARD_COLS, BOARD_ROWS } from './board.ts';

export const NEIGHBOR_DIRS = POINTY_DIRECTIONS.length; // 6

/** Axial (dq, dr) delta per direction, POINTY_DIRECTIONS order. */
const AXIAL_DELTAS: ReadonlyArray<readonly [number, number]> = (() => {
    // A 3x3 scratch grid; its center hex has all 6 neighbors in-grid.
    const grid = new Grid(HexTile, rectangle({ width: 3, height: 3 }));
    let center: HexTile | undefined;
    grid.forEach((hex) => {
        if (hex.row === 1 && hex.col === 1) center = hex;
    });
    return POINTY_DIRECTIONS.map((dir) => {
        const nb = grid.neighborOf(center!, dir, { allowOutside: false })!;
        return [nb.q - center!.q, nb.r - center!.r] as const;
    });
})();

/** [cell * NEIGHBOR_DIRS + d] → neighbor cell index, or -1 off-window. */
export const NEIGHBOR_INDEX: Int32Array = (() => {
    const table = new Int32Array(BOARD_CELLS * NEIGHBOR_DIRS).fill(-1);
    for (let row = 0; row < BOARD_ROWS; row++) {
        for (let col = 0; col < BOARD_COLS; col++) {
            const c = row * BOARD_COLS + col;
            for (let d = 0; d < NEIGHBOR_DIRS; d++) {
                const [dq, dr] = AXIAL_DELTAS[d];
                const nCol = col + dq;
                const nRow = row + dr;
                if (nCol >= 0 && nCol < BOARD_COLS && nRow >= 0 && nRow < BOARD_ROWS) {
                    table[c * NEIGHBOR_DIRS + d] = nRow * BOARD_COLS + nCol;
                }
            }
        }
    }
    return table;
})();
