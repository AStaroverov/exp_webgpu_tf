/**
 * A* pathfinding over the hex grid.
 *
 * honeycomb-grid provides geometry/neighbors/distance but no pathfinder, so this
 * is a thin A* on top of `HexGrid`: neighbors come from the grid, the heuristic
 * is the exact hex distance (admissible, so A* is optimal), and blocked cells are
 * skipped dynamically — no graph rebuild needed when occupancy changes.
 */

import type { HexCoordinates } from "honeycomb-grid";
import { cellKey, HexGrid } from "./HexGrid.ts";

export type PathHex = { q: number; r: number };

export type FindPathOptions = {
  /**
   * Override the default "can step onto this cell" test. By default a cell is
   * blocked if it is not passable (missing / non-walkable / occupied). The
   * start cell is always allowed regardless of this test.
   */
  isBlocked?: (q: number, r: number) => boolean;
};

/** Minimal binary min-heap keyed by fScore. */
class MinHeap {
  private heap: Array<{ k: string; f: number }> = [];

  get size(): number {
    return this.heap.length;
  }

  push(k: string, f: number): void {
    const h = this.heap;
    h.push({ k, f });
    let i = h.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (h[parent].f <= h[i].f) break;
      [h[parent], h[i]] = [h[i], h[parent]];
      i = parent;
    }
  }

  pop(): string | undefined {
    const h = this.heap;
    if (h.length === 0) return undefined;
    const top = h[0];
    const last = h.pop()!;
    if (h.length > 0) {
      h[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        let smallest = i;
        if (l < h.length && h[l].f < h[smallest].f) smallest = l;
        if (r < h.length && h[r].f < h[smallest].f) smallest = r;
        if (smallest === i) break;
        [h[smallest], h[i]] = [h[i], h[smallest]];
        i = smallest;
      }
    }
    return top.k;
  }
}

/**
 * Find a shortest hex path from `start` to `goal` (inclusive of both).
 * Returns null when no path exists. Each step has uniform cost 1.
 */
export function findPath(
  grid: HexGrid,
  start: PathHex,
  goal: PathHex,
  options: FindPathOptions = {},
): PathHex[] | null {
  if (!grid.has(start as HexCoordinates) || !grid.has(goal as HexCoordinates)) {
    return null;
  }

  const isBlocked = options.isBlocked ?? ((q, r) => !grid.isPassable(q, r));

  const startKey = cellKey(start.q, start.r);
  const goalKey = cellKey(goal.q, goal.r);

  const cameFrom = new Map<string, PathHex>();
  const gScore = new Map<string, number>([[startKey, 0]]);
  const open = new MinHeap();
  const closed = new Set<string>();

  open.push(startKey, grid.distance(start as HexCoordinates, goal as HexCoordinates));
  const coords = new Map<string, PathHex>([[startKey, start]]);

  while (open.size > 0) {
    const currentKey = open.pop()!;
    if (currentKey === goalKey) {
      return reconstruct(cameFrom, goal, startKey);
    }
    if (closed.has(currentKey)) continue;
    closed.add(currentKey);

    const current = coords.get(currentKey)!;
    const g = gScore.get(currentKey)!;

    for (const n of grid.neighbors(current as HexCoordinates)) {
      const nKey = cellKey(n.q, n.r);
      if (closed.has(nKey)) continue;
      // The start cell is always steppable; otherwise apply the block test.
      if (nKey !== startKey && isBlocked(n.q, n.r)) continue;

      const tentative = g + 1;
      if (tentative < (gScore.get(nKey) ?? Infinity)) {
        cameFrom.set(nKey, current);
        gScore.set(nKey, tentative);
        coords.set(nKey, { q: n.q, r: n.r });
        const f = tentative + grid.distance(n, goal as HexCoordinates);
        open.push(nKey, f);
      }
    }
  }

  return null;
}

function reconstruct(cameFrom: Map<string, PathHex>, goal: PathHex, startKey: string): PathHex[] {
  const path: PathHex[] = [goal];
  let curKey = cellKey(goal.q, goal.r);
  while (curKey !== startKey) {
    const prev = cameFrom.get(curKey);
    if (!prev) break;
    path.push(prev);
    curKey = cellKey(prev.q, prev.r);
  }
  path.reverse();
  return path;
}
