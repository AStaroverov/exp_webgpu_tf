/**
 * Grid occupancy — the single writer of the hex grid's *dynamic* occupancy layer.
 *
 * Occupancy is a pure projection of vehicle state, rebuilt from scratch every tick
 * (run right after the physics step, before actions/decisions read the grid). No
 * other system mutates Unit/Reserved cells — MoveStep and friends only *read* the
 * grid, so there is no scattered occupy/vacate to leak a stale reservation.
 *
 * Each tick:
 *   1. clear every Unit/Reserved cell (static Obstacles are left untouched);
 *   2. mark the cell each vehicle currently sits on as Unit (occupied);
 *   3. mark the cell each *moving* vehicle is heading into as Reserved — derived
 *      from its position + velocity (one cell ahead), skipping cells that are
 *      already taken.
 */

import { query } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { MapDI } from "../../../DI/MapDI.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import { OccupantKind } from "../../../Map/HexGrid.ts";
import { HexGridConfig } from "../../../Map/HexConfig.ts";

/** Speed (world units/tick) below which a vehicle is treated as not moving (no reservation). */
const MIN_SPEED = 0.5;
/** How far ahead (world units) along velocity we look for the cell being entered. */
const LOOKAHEAD = HexGridConfig.radius;

export function createGridOccupancySystem({ world } = GameDI) {
  const { Vehicle, RigidBodyState } = getGameComponents(world);

  return function updateGridOccupancy() {
    const grid = MapDI.grid;
    if (!grid) return;

    // 1) Clear the dynamic layer; keep static obstacles.
    grid.forEachCell((cell) => {
      if (cell.occupantKind === OccupantKind.Unit || cell.occupantKind === OccupantKind.Reserved) {
        grid.vacate(cell.q, cell.r);
      }
    });

    const vehicles = query(world, [Vehicle, RigidBodyState]);

    // 2) Occupied cells — where each vehicle physically sits now.
    for (const eid of vehicles) {
      const px = RigidBodyState.position.get(eid, 0);
      const py = RigidBodyState.position.get(eid, 1);
      const here = grid.worldToHex(px, py);
      if (here) grid.occupy(here.q, here.r, eid, OccupantKind.Unit);
    }

    // 3) Reserved cells — the cell each moving vehicle is driving into.
    for (const eid of vehicles) {
      const vx = RigidBodyState.linvel.get(eid, 0);
      const vy = RigidBodyState.linvel.get(eid, 1);
      const speed = Math.hypot(vx, vy);
      if (speed < MIN_SPEED) continue;

      const px = RigidBodyState.position.get(eid, 0);
      const py = RigidBodyState.position.get(eid, 1);
      const ahead = grid.worldToHex(px + (vx / speed) * LOOKAHEAD, py + (vy / speed) * LOOKAHEAD);
      if (!ahead) continue;
      // Don't overwrite a cell that's already taken (the vehicle's own current
      // cell, another unit, an obstacle, or someone else's earlier reservation).
      if (grid.getOccupant(ahead.q, ahead.r) !== null) continue;
      grid.occupy(ahead.q, ahead.r, eid, OccupantKind.Reserved);
    }
  };
}
