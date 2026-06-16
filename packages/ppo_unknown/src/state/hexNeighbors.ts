/**
 * Hex topology of the EGOCENTRIC board window (see board.ts): the flat-index
 * neighbor table plus the fire-target ring offsets. The window is indexed by
 * AXIAL deltas (col = dq + R, row = dr + R), and axial neighbor offsets are
 * parity-free constants — so the tables are uniform shifts, -1 at the window
 * edge. Direction order = POINTY_DIRECTIONS, the same order the move action
 * slice uses (see consts.ts).
 *
 * The per-direction axial deltas are derived from honeycomb itself (not
 * hardcoded) so they can never drift from the game's hex topology.
 *
 * Used graph-side (the gather layers' constant matrices; only the center
 * cell's row is actually read there — self sits at the window center) and
 * decision-side (`applyActionToGame` / `computeActionMask` resolve a fire
 * action index to its target hex through `FIRE_TARGET_OFFSETS`).
 */

import { Grid, rectangle } from "honeycomb-grid";
import { HexTile, POINTY_DIRECTIONS } from "../../../unknown/src/Game/Map/HexConfig.ts";
import { BOARD_CELLS, BOARD_COLS, BOARD_ROWS, VIEW_RADIUS } from "./board.ts";

export const NEIGHBOR_DIRS = POINTY_DIRECTIONS.length; // 6

/** Axial (dq, dr) delta per direction, POINTY_DIRECTIONS order. */
export const AXIAL_DELTAS: ReadonlyArray<readonly [number, number]> = (() => {
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

/**
 * Fire actions target a precise hex within this many rings around the tank
 * (mechanics: "не 6 направлений, а 3 радиуса вокруг танка"). Must stay ≤
 * VIEW_RADIUS so every fire target is an observed board cell.
 */
export const FIRE_RING_RADIUS = 3;

/**
 * Axial (dq, dr) offset of every fire-target hex, rings 1..FIRE_RING_RADIUS
 * walked ring by ring (6, 12, 18 cells → 36 total). Within ring k, side `i`
 * starts at the corner `k × AXIAL_DELTAS[i]` and walks its k cells toward the
 * next corner — so ring 1 is exactly AXIAL_DELTAS (POINTY_DIRECTIONS order, the
 * same cells the move slice steps into). One fire action slot per entry; the
 * order is the single source of truth shared by the action layout
 * (consts.ts), the decision seam, and the network's ring gather.
 */
export const FIRE_TARGET_OFFSETS: ReadonlyArray<readonly [number, number]> = (() => {
  const offsets: Array<readonly [number, number]> = [];
  for (let k = 1; k <= FIRE_RING_RADIUS; k++) {
    for (let side = 0; side < NEIGHBOR_DIRS; side++) {
      const [cq, cr] = AXIAL_DELTAS[side];
      // Stepping from corner i toward corner i+1 follows direction i+2 (hex ring walk).
      const [sq, sr] = AXIAL_DELTAS[(side + 2) % NEIGHBOR_DIRS];
      for (let step = 0; step < k; step++) {
        offsets.push([k * cq + step * sq, k * cr + step * sr] as const);
      }
    }
  }
  return offsets;
})();

/**
 * Window cell index of every ACTION CELL: self (the window center — the board
 * is egocentric, see board.ts) followed by each fire-target ring cell
 * (FIRE_TARGET_OFFSETS order). All in-window by construction
 * (FIRE_RING_RADIUS ≤ VIEW_RADIUS, asserted). Fed to the network's gather
 * layer (v4) — the latents ARE these cells' tokens.
 */
export const ACTION_CELL_INDEXES: readonly number[] = (() => {
  const indexes = [VIEW_RADIUS * BOARD_COLS + VIEW_RADIUS];
  for (const [dq, dr] of FIRE_TARGET_OFFSETS) {
    const col = VIEW_RADIUS + dq;
    const row = VIEW_RADIUS + dr;
    if (col < 0 || col >= BOARD_COLS || row < 0 || row >= BOARD_ROWS) {
      throw new Error(`fire target offset (${dq}, ${dr}) falls outside the board window`);
    }
    indexes.push(row * BOARD_COLS + col);
  }
  return indexes;
})();

/**
 * Axial (dq, dr) offset of EVERY board-window cell, indexed by its flat cell
 * index (`row * BOARD_COLS + col`) — the inverse of the egocentric window
 * mapping (col = dq + R, row = dr + R). This is the fire-target table for the
 * full-board aim head (v5): the policy may fire at ANY reachable window cell,
 * so a fire action index IS a cell index, resolved to a hex through this table.
 * The window center (cell `VIEW_RADIUS*BOARD_COLS + VIEW_RADIUS`) maps to
 * (0, 0) — self, never a valid fire target.
 */
export const FIRE_CELL_OFFSETS: ReadonlyArray<readonly [number, number]> = (() => {
  const offsets: Array<readonly [number, number]> = [];
  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      offsets.push([col - VIEW_RADIUS, row - VIEW_RADIUS] as const);
    }
  }
  return offsets;
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
