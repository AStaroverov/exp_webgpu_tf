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
 *   3. mark every free neighbor of an Obstacle cell as Reserved (buffer ring);
 *   4. mark every free neighbor of each vehicle's cell as Reserved (buffer ring).
 *
 * A buffer ring is reserved with its owner's eid, so a vehicle may still enter
 * its *own* ring (that's how it leaves its cell) while everyone else treats the
 * ring as blocked. A cell sitting in TWO rings (adjacent to both owners) is
 * contested: it is re-reserved with no owner (eid 0), so neither vehicle may
 * enter it — otherwise whoever reserved it first could drive right up to the
 * other one.
 */

import { EntityId, query } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { MapDI } from "../../../DI/MapDI.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import { HexCell, HexGrid, OccupantKind } from "../../../Map/HexGrid.ts";

export function createGridOccupancySystem({ world } = GameDI) {
  const { Vehicle, RigidBodyState } = getGameComponents(world);
  const obstacleCells: HexCell[] = [];

  return function updateGridOccupancy() {
    const grid = MapDI.grid;
    if (!grid) return;

    // 1) Clear the dynamic layer; collect static obstacles for their buffer ring.
    obstacleCells.length = 0;
    grid.forEachCell((cell) => {
      if (cell.occupantKind === OccupantKind.Unit || cell.occupantKind === OccupantKind.Reserved) {
        grid.vacate(cell.q, cell.r);
      } else if (cell.occupantKind === OccupantKind.Obstacle) {
        obstacleCells.push(cell);
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

    // 3) Buffer ring around every obstacle.
    for (const cell of obstacleCells) {
      reserveNeighbors(grid, cell.q, cell.r, cell.occupantEid!);
    }

    // 4) Buffer ring around every vehicle.
    for (const eid of vehicles) {
      const px = RigidBodyState.position.get(eid, 0);
      const py = RigidBodyState.position.get(eid, 1);
      const here = grid.worldToHex(px, py);
      if (here) reserveNeighbors(grid, here.q, here.r, eid);
    }
  };
}

/**
 * No entity — a contested ring cell (adjacent to two different owners) is
 * re-reserved with this sentinel so `isPassableFor` blocks it for EVERYONE:
 * whoever reserved it first must not get to drive right up to the other owner.
 */
const NO_OWNER: EntityId = 0 as EntityId;

/** Reserve every free neighbor of (q, r) on behalf of `eid`; contested cells lose their owner. */
function reserveNeighbors(grid: HexGrid, q: number, r: number, eid: EntityId): void {
  for (const n of grid.neighbors({ q, r })) {
    const occ = grid.getOccupant(n.q, n.r);
    if (occ === null) {
      grid.occupy(n.q, n.r, eid, OccupantKind.Reserved);
    } else if (occ.kind === OccupantKind.Reserved && occ.eid !== eid) {
      grid.occupy(n.q, n.r, NO_OWNER, OccupantKind.Reserved);
    }
  }
}
