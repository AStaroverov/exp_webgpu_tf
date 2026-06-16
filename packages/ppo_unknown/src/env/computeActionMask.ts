/**
 * computeActionMask — build the per-decision invalid-action mask for one agent.
 *
 * The mask is a flat `Float32Array` of length `ACTION_DIM_TOTAL` (128), laid out
 * identically to the single flat policy head (Hold | move | fire) and additive:
 *   `0`        → action allowed,
 *   `MASK_NEG` → action forbidden (added to the logit before softmax/argmax).
 *
 * Slices (see consts.ts action layout):
 *   Hold [0]            — never masked, so the distribution always has a valid action.
 *   move [1..6]         — `0` for each passable hex neighbour (shared predicate with
 *                         `applyActionToGame.moveDestination`), `MASK_NEG` otherwise.
 *   fire [7..7+CELLS)   — `MASK_NEG` for the WHOLE slice while the agent can't shoot
 *                         (`Firearms` reloading, or `StreamFirearms` charge below the
 *                         firing threshold — a shot is impossible anyway); otherwise per WINDOW CELL
 *                         (`FIRE_CELL_OFFSETS`): `MASK_NEG` for the self cell, a cell
 *                         beyond the view radius (the window corners), or an off-grid
 *                         hex; `0` for any reachable on-grid cell. Firing at an empty
 *                         hex just wastes the shot — the reward, not the mask, discourages that.
 */

import { hasComponent } from "bitecs";
import { GameDI } from "../../../unknown/src/Game/DI/GameDI.ts";
import { MapDI } from "../../../unknown/src/Game/DI/MapDI.ts";
import { getGameComponents } from "../../../unknown/src/Game/ECS/createGameWorld.ts";
import { VIEW_RADIUS, hexDeltaDistance } from "../state/board.ts";
import {
  ACTION_DIM_TOTAL,
  ACTION_GROUP_PRIOR,
  FIRE_ACTION_OFFSET,
  FIRE_CELL_OFFSETS,
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

  if (!here) return mask;

  for (let dir = 0; dir < MOVE_DIR_COUNT; dir++) {
    if (!moveDestination(grid, eid, here.q, here.r, dir)) {
      mask[MOVE_ACTION_OFFSET + dir] = MASK_NEG;
    }
  }

  // ── fire slice [7..42] ────────────────────────────────────────────────────
  const turretEid = Tank.turretEId.get(eid);
  // Bullet guns: reloading. Stream guns: charge below the firing threshold.
  // Either way a shot is impossible, so the whole fire slice is dead.
  const gunBlocked =
    turretEid !== 0 &&
    ((hasComponent(world, turretEid, Firearms) && Firearms.isReloading(turretEid)) ||
      (hasComponent(world, turretEid, StreamFirearms) && !StreamFirearms.canFire(turretEid)));
  for (let i = 0; i < FIRE_TARGET_COUNT; i++) {
    // While reloading the whole slice is dead (a shot is impossible anyway).
    // Otherwise a cell is a valid target iff it is a real on-grid hex other
    // than self — the policy may fire at ANY reachable window cell.
    const [dq, dr] = FIRE_CELL_OFFSETS[i];
    const isSelf = dq === 0 && dr === 0;
    const inView = hexDeltaDistance(dq, dr) <= VIEW_RADIUS; // exclude the window corners
    const onGrid = inView && grid.has({ q: here.q + dq, r: here.r + dr });
    if (gunBlocked || isSelf || !onGrid) {
      mask[FIRE_ACTION_OFFSET + i] = MASK_NEG;
    }
  }

  if (mask.subarray(1).every((v) => v === MASK_NEG)) {
    mask[0] = 0;
  } else {
    mask[0] = MASK_NEG;
  }
  return mask;
}

/**
 * The additive logit vector for the TRAINED policy: the validity mask PLUS the
 * constant per-group exploration prior (`ACTION_GROUP_PRIOR`). Forbidden entries
 * stay ≈ `MASK_NEG` (the small prior is dwarfed by -1e9); allowed entries carry
 * their group's prior instead of 0. This is the vector both sampled-with AND
 * stored for the loss, so old/new logprob include the same prior and the PPO
 * ratio stays consistent. Use this for the learned agents (UnknownAgent /
 * FrozenAgent) — NOT for `RandomBot`, which reads the raw 0/MASK_NEG validity
 * mask as a boolean ("allowed iff === 0").
 */
export function computeActionMaskWithPrior(eid: number, di = GameDI): Float32Array {
  const mask = computeActionMask(eid, di);
  for (let i = 0; i < mask.length; i++) {
    mask[i] += ACTION_GROUP_PRIOR[i];
  }
  return mask;
}
