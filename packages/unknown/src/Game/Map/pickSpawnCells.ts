/**
 * pickSpawnCells — choose unit spawn cells from a candidate pool.
 *
 * Walks `pool` in order (pre-shuffle / pre-sort it for the placement you want)
 * and greedily accepts cells that are valid spawn spots:
 *   - passable (no unit/obstacle/reservation on the cell);
 *   - not adjacent to an obstacle (the cell would sit inside its buffer ring);
 *   - not adjacent to any already-accepted cell, nor to any cell in `picked` —
 *     units never spawn inside each other's buffer ring.
 *
 * Pure grid logic — no ECS access; pass previously chosen cells via `picked`
 * to keep the no-adjacency rule across several calls (e.g. per-team picking).
 */

import { HexGrid, OccupantKind } from "./HexGrid.ts";

type Cell = { q: number; r: number };

export function pickSpawnCells(
  grid: HexGrid,
  pool: ReadonlyArray<Cell>,
  count: number,
  picked: ReadonlyArray<Cell> = [],
): Cell[] {
  const out: Cell[] = [];
  for (const cell of pool) {
    if (out.length >= count) break;
    if (!grid.isPassable(cell.q, cell.r)) continue;
    if (touchesObstacle(grid, cell)) continue;
    if (touchesAny(grid, cell, picked) || touchesAny(grid, cell, out)) continue;
    out.push({ q: cell.q, r: cell.r });
  }
  return out;
}

/** True when the cell has a static obstacle in its neighbor ring. */
function touchesObstacle(grid: HexGrid, cell: Cell): boolean {
  for (const n of grid.neighbors(cell)) {
    if (grid.getOccupant(n.q, n.r)?.kind === OccupantKind.Obstacle) return true;
  }
  return false;
}

/** True when the cell coincides with or is adjacent to any cell in `cells`. */
function touchesAny(grid: HexGrid, cell: Cell, cells: ReadonlyArray<Cell>): boolean {
  for (const other of cells) {
    if (grid.distance(cell, other) < 2) return true;
  }
  return false;
}
