/**
 * computeActionMask — build the per-decision invalid-action mask for one agent.
 *
 * The mask is a flat `Float32Array` of length `ACTION_DIM_TOTAL` (13), laid out
 * identically to the single flat policy head (Hold | move | fire) and additive:
 *   `0`        → action allowed,
 *   `MASK_NEG` → action forbidden (added to the logit before softmax/argmax).
 *
 * Slices (see consts.ts action layout):
 *   Hold [0]      — never masked, so the distribution always has a valid action.
 *   move [1..6]   — `0` for each passable hex neighbour (shared predicate with
 *                   `applyActionToGame.moveDestination`), `MASK_NEG` otherwise.
 *   fire [7..12]  — `MASK_NEG` for each direction whose line-of-fire hits a friendly
 *                   Unit before any enemy (no friendly fire), `0` otherwise. Firing
 *                   down an empty line just wastes the shot — the reward, not the
 *                   mask, discourages that.
 */

import { GameDI } from "../../../unknown/src/Game/DI/GameDI.ts";
import { MapDI } from "../../../unknown/src/Game/DI/MapDI.ts";
import { getGameComponents } from "../../../unknown/src/Game/ECS/createGameWorld.ts";
import { OccupantKind } from "../../../unknown/src/Game/Map/HexGrid.ts";
import {
  ACTION_DIM_TOTAL,
  FIRE_ACTION_OFFSET,
  FIRE_DIR_COUNT,
  MASK_NEG,
  MOVE_ACTION_OFFSET,
  MOVE_DIR_COUNT,
} from "../consts.ts";
import { moveDestination } from "./applyActionToGame.ts";

export function computeActionMask(eid: number, { world } = GameDI): Float32Array {
  const mask = new Float32Array(ACTION_DIM_TOTAL); // all 0 (every action allowed by default)
  const grid = MapDI.grid;
  if (!grid) return mask; // no grid → nothing to forbid

  const { RigidBodyState, TeamRef } = getGameComponents(world);

  // ── move slice [1..6] ─────────────────────────────────────────────────────
  const px = RigidBodyState.position.get(eid, 0);
  const py = RigidBodyState.position.get(eid, 1);
  const here = grid.worldToHex(px, py);
  if (here) {
    for (let dir = 0; dir < MOVE_DIR_COUNT; dir++) {
      if (!moveDestination(grid, here.q, here.r, dir)) {
        mask[MOVE_ACTION_OFFSET + dir] = MASK_NEG;
      }
    }
  }
  // if `here` is undefined we leave the move slice all 0 (no info → don't forbid).

  // ── fire slice [7..12] ────────────────────────────────────────────────────
  // The projectile travels along the whole direction line, so we walk the ray
  // from the firing cell outward and look at the FIRST blocker it hits:
  //   • friendly Unit first  → forbid (would hit an ally before any enemy);
  //   • enemy Unit / Obstacle first → allow (the round stops on it, no friendly fire);
  //   • Reserved / empty cells are passed through (bullet flies over them).
  // Empty rays stay fireable (0) — a wasted shot is discouraged by the reward,
  // not the mask. The first ray step uses `neighborAt(here, dir)` (direction-stable
  // slot, matching `applyActionToGame`); subsequent steps repeat that axial delta
  // (a straight hex line).
  if (here) {
    const myTeam = TeamRef.id[eid];
    for (let dir = 0; dir < FIRE_DIR_COUNT; dir++) {
      const target = grid.neighborAt(here, dir);
      if (!target) continue; // off-grid direction → no-op, leave allowed (0)
      const dq = target.q - here.q;
      const dr = target.r - here.r;
      let q = target.q;
      let r = target.r;
      let blockedByAlly = false;
      // walk the straight hex line until we leave the grid or hit a blocker
      while (grid.getCell(q, r)) {
        const occupant = grid.getOccupant(q, r);
        if (occupant) {
          if (occupant.kind === OccupantKind.Unit) {
            blockedByAlly = TeamRef.id[occupant.eid] === myTeam;
            break; // first unit hit: ally → forbid, enemy → allow
          }
          if (occupant.kind === OccupantKind.Obstacle) break; // bullet stops, no ally hit
          // OccupantKind.Reserved: physically empty → bullet flies over
        }
        q += dq;
        r += dr;
      }
      if (blockedByAlly) {
        mask[FIRE_ACTION_OFFSET + dir] = MASK_NEG;
      }
    }
  }

  return mask;
}
