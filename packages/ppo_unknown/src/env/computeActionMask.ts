/**
 * computeActionMask — build the per-decision invalid-action mask for one agent.
 *
 * The mask is a flat `Float32Array` of length `ACTION_DIM_TOTAL` (43), laid out
 * identically to the single flat policy head (Hold | move | fire) and additive:
 *   `0`        → action allowed,
 *   `MASK_NEG` → action forbidden (added to the logit before softmax/argmax).
 *
 * Slices (see consts.ts action layout):
 *   Hold [0]      — never masked, so the distribution always has a valid action.
 *   move [1..6]   — `0` for each passable hex neighbour (shared predicate with
 *                   `applyActionToGame.moveDestination`), `MASK_NEG` otherwise.
 *   fire [7..42]  — `MASK_NEG` for the WHOLE slice while the agent's gun is reloading
 *                   (`Firearms` or `StreamFirearms` — a shot is impossible anyway);
 *                   otherwise per TARGET HEX (rings 1..FIRE_RING_RADIUS,
 *                   `FIRE_TARGET_OFFSETS` — same table as `applyActionToGame`):
 *                   `MASK_NEG` when the hex is off-grid (not a real target), or when
 *                   the line of fire toward it hits a friendly Unit before any enemy
 *                   (no friendly fire); `0` otherwise. Firing at an empty hex just
 *                   wastes the shot — the reward, not the mask, discourages that.
 */

import { hasComponent } from "bitecs";
import { GameDI } from "../../../unknown/src/Game/DI/GameDI.ts";
import { MapDI } from "../../../unknown/src/Game/DI/MapDI.ts";
import { getGameComponents } from "../../../unknown/src/Game/ECS/createGameWorld.ts";
import {
  ACTION_DIM_TOTAL,
  FIRE_ACTION_OFFSET,
  FIRE_TARGET_COUNT,
  MASK_NEG,
  MOVE_ACTION_OFFSET,
  MOVE_DIR_COUNT,
} from "../consts.ts";
import { moveDestination } from "./applyActionToGame.ts";

export function computeActionMask(eid: number, { world } = GameDI): Float32Array {
  const mask = new Float32Array(ACTION_DIM_TOTAL); // all 0 (every action allowed by default)
  const grid = MapDI.grid;
  if (!grid) return mask; // no grid → nothing to forbid

  const { RigidBodyState, Tank, Firearms, StreamFirearms } = getGameComponents(world);

  // ── move slice [1..6] ─────────────────────────────────────────────────────
  const px = RigidBodyState.position.get(eid, 0);
  const py = RigidBodyState.position.get(eid, 1);
  const here = grid.worldToHex(px, py)!;

  for (let dir = 0; dir < MOVE_DIR_COUNT; dir++) {
    if (!moveDestination(grid, here.q, here.r, dir)) {
      mask[MOVE_ACTION_OFFSET + dir] = MASK_NEG;
    }
  }

  // ── fire slice [7..42] ────────────────────────────────────────────────────
  const turretEid = Tank.turretEId[eid];
  const gunReloading =
    turretEid !== 0 &&
    ((hasComponent(world, turretEid, Firearms) && Firearms.isReloading(turretEid)) ||
      (hasComponent(world, turretEid, StreamFirearms) && StreamFirearms.isReloading(turretEid)));
  if (gunReloading) {
    for (let i = 0; i < FIRE_TARGET_COUNT; i++) {
      mask[FIRE_ACTION_OFFSET + i] = MASK_NEG;
    }
    return mask;
  }

  return mask;
}
