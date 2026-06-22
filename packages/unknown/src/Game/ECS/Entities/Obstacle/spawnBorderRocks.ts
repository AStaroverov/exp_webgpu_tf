/**
 * spawnBorderRocks — line the grid's outer contour with rocks (a wall ring),
 * framing the arena. A cell sits on the border when it has fewer than 6 in-grid
 * neighbours (the rectangle's edge), so we drop one rock on each such cell.
 *
 * Border cells are collected first, THEN committed, so we never mutate grid
 * occupancy while iterating it. The interior stays fully connected (the ring is
 * the outer shell), so it composes with the random interior `spawnObstacles`.
 */

import { GameDI } from "../../../DI/GameDI.ts";
import { MapDI } from "../../../DI/MapDI.ts";
import { createRock } from "./Rocks/Rock.ts";

export function spawnBorderRocks({ grid } = MapDI, { world } = GameDI): void {
  const border: Array<{ q: number; r: number }> = [];
  grid.forEachCell((cell) => {
    if (grid.neighbors({ q: cell.q, r: cell.r }).length < 6) {
      border.push({ q: cell.q, r: cell.r });
    }
  });

  for (const cell of border) {
    createRock({ anchor: cell, cells: [cell] }, { world });
  }
}
