/**
 * Flat board-index hex neighbor table, built once from the game's hex topology
 * (honeycomb, row-parity offsets included). Direction order = POINTY_DIRECTIONS,
 * the same order the move/fire action slices use (see consts.ts).
 *
 * Used CPU-side (content mask in InputTensors) and graph-side
 * (HexNeighborGatherLayer's constant gather matrix).
 */

import { Grid, rectangle } from 'honeycomb-grid';
import { HexGridConfig, HexTile, POINTY_DIRECTIONS } from '../../../unknown/src/Game/Map/HexConfig.ts';
import { BOARD_CELLS, BOARD_COLS } from './board.ts';

export const NEIGHBOR_DIRS = POINTY_DIRECTIONS.length; // 6

/** [cell * NEIGHBOR_DIRS + d] → neighbor cell index, or -1 off-board. */
export const NEIGHBOR_INDEX: Int32Array = (() => {
    const table = new Int32Array(BOARD_CELLS * NEIGHBOR_DIRS).fill(-1);
    const grid = new Grid(HexTile, rectangle({ width: HexGridConfig.cols, height: HexGridConfig.rows }));
    grid.forEach((hex) => {
        const c = hex.row * BOARD_COLS + hex.col;
        for (let d = 0; d < NEIGHBOR_DIRS; d++) {
            const nb = grid.neighborOf(hex, POINTY_DIRECTIONS[d], { allowOutside: false });
            if (nb) table[c * NEIGHBOR_DIRS + d] = nb.row * BOARD_COLS + nb.col;
        }
    });
    return table;
})();
