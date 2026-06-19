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
 *                         firing threshold — a shot is impossible anyway); otherwise a
 *                         cell is allowed (`0`) ONLY when it both (a) holds an ENEMY —
 *                         an enemy unit, or a reserved buffer cell around one (aiming at
 *                         the ring still resolves onto the enemy, see
 *                         `FireAction.resolveTargetEid`) — and (b) sits within the gun's
 *                         firing reach. Empty hexes, allies, obstacles, and cells beyond
 *                         range/view are all `MASK_NEG`.
 */

import { hasComponent } from "bitecs";
import { GameDI } from "../../../unknown/src/Game/DI/GameDI.ts";
import { MapDI } from "../../../unknown/src/Game/DI/MapDI.ts";
import { getGameComponents } from "../../../unknown/src/Game/ECS/createGameWorld.ts";
import { OccupantKind } from "../../../unknown/src/Game/Map/HexGrid.ts";
import { HexGridConfig } from "../../../unknown/src/Game/Map/HexConfig.ts";
import {
  BulletCaliberConfig,
  StreamCaliberConfig,
} from "../../../unknown/src/Game/Config/weapons.ts";
import { VIEW_RADIUS, hexDeltaDistance } from "../state/board.ts";
import {
  ACTION_DIM_TOTAL,
  ACTION_GROUP_TARGET_SHARE,
  FIRE_ACTION_OFFSET,
  FIRE_CELL_OFFSETS,
  FIRE_TARGET_COUNT,
  HOLD_ACTION,
  MASK_NEG,
  MOVE_ACTION_OFFSET,
  MOVE_DIR_COUNT,
} from "../consts.ts";
import { moveDestination } from "./applyActionToGame.ts";

/**
 * World-space distance between two adjacent pointy-top hex centers (`√3 · radius`).
 * Converts a weapon's world-unit reach into hex steps so the fire mask can compare
 * it against `hexDeltaDistance`.
 */
const HEX_STEP_WORLD = Math.sqrt(3) * HexGridConfig.radius;

export function computeActionMask(eid: number, { world } = GameDI): Float32Array {
  const mask = new Float32Array(ACTION_DIM_TOTAL); // all 0 (every action allowed by default)
  const grid = MapDI.grid;
  if (!grid) return mask; // no grid → nothing to forbid

  const { RigidBodyState, Tank, Firearms, StreamFirearms, TeamRef, Vehicle } =
    getGameComponents(world);

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

  // ── fire slice [7..] ──────────────────────────────────────────────────────
  const turretEid = Tank.turretEId.get(eid);
  const myTeamId = TeamRef.id.get(eid);

  // Bullet guns: reloading. Stream guns: charge below the firing threshold.
  // Either way a shot is impossible, so the whole fire slice is dead. A gunless
  // turret (eid 0) also can't fire.
  const gunBlocked =
    turretEid === 0 ||
    (hasComponent(world, turretEid, Firearms) && Firearms.isReloading(turretEid)) ||
    (hasComponent(world, turretEid, StreamFirearms) && !StreamFirearms.canFire(turretEid));

  // The gun's reach in hex steps — cells farther than this can't be hit. Bullet
  // guns: the caliber's hard `maxDistance` travel cutoff. Stream guns: the
  // particles' damped flight reach `v/k · (1 − e^(−k·t))` (speed `v`, damping `k`,
  // lifetime `t`). Both converted from world units to hex steps.
  let rangeSteps = 0;
  if (!gunBlocked) {
    if (hasComponent(world, turretEid, Firearms)) {
      rangeSteps = Math.round(
        BulletCaliberConfig[Firearms.getCaliber(turretEid)].maxDistance / HEX_STEP_WORLD,
      );
    } else if (hasComponent(world, turretEid, StreamFirearms)) {
      const cfg = StreamCaliberConfig[StreamFirearms.caliberRef.get(turretEid)];
      const reach =
        (cfg.speed / cfg.linearDamping) *
        (1 - Math.exp((-cfg.linearDamping * cfg.lifetimeMs) / 1000));
      rangeSteps = Math.max(1, Math.round(reach / HEX_STEP_WORLD));
    }
  }

  for (let i = 0; i < FIRE_TARGET_COUNT; i++) {
    // A cell is a valid target ONLY when it is in reach AND holds an enemy — an
    // enemy unit, or a reserved buffer cell owned by an enemy (aiming at the ring
    // still resolves onto the enemy). `getCell` returns the stored cell (no alloc).
    const [dq, dr] = FIRE_CELL_OFFSETS[i];
    const d = hexDeltaDistance(dq, dr);
    let allowed = !gunBlocked && d >= 1 && d <= VIEW_RADIUS && d <= rangeSteps;
    if (allowed) {
      const cell = grid.getCell(here.q + dq, here.r + dr);
      allowed =
        cell != null &&
        cell.occupantEid !== null &&
        cell.occupantEid !== 0 && // NO_OWNER: a contested reserved ring, no known owner
        cell.occupantKind !== OccupantKind.Obstacle &&
        hasComponent(world, cell.occupantEid, Vehicle) && // an enemy obstacle's ring is owned by a non-vehicle
        TeamRef.id.get(cell.occupantEid) !== myTeamId;
    }
    if (!allowed) {
      mask[FIRE_ACTION_OFFSET + i] = MASK_NEG;
    }
  }

  return mask;
}

// The three action groups, as [start, count] spans into the flat mask, paired with
// their init target share. Hold is a single never-masked slot; move and fire are the
// maskable slices. Used by the exploration prior below.
const PRIOR_GROUPS: ReadonlyArray<{ start: number; count: number; share: number }> = [
  { start: HOLD_ACTION, count: 1, share: ACTION_GROUP_TARGET_SHARE.hold },
  { start: MOVE_ACTION_OFFSET, count: MOVE_DIR_COUNT, share: ACTION_GROUP_TARGET_SHARE.move },
  { start: FIRE_ACTION_OFFSET, count: FIRE_TARGET_COUNT, share: ACTION_GROUP_TARGET_SHARE.fire },
];

/**
 * The additive logit vector for the TRAINED policy: the validity mask PLUS a per-group
 * exploration prior. The prior spreads each group's `ACTION_GROUP_TARGET_SHARE` over
 * the actions that are VALID this tick (`log(share / validCountInGroup)`), so with the
 * network's raw logits ≈0 at init the masked softmax realizes the target shares
 * (hold .10 / move .45 / fire .45) EXACTLY, in every state — not only when a group is
 * fully unmasked. A group with no valid action this tick (e.g. fire while reloading)
 * is left alone; its entries stay ≈`MASK_NEG` and its share flows to the live groups.
 *
 * This is the vector both sampled-with AND stored for the loss, so old/new logprob
 * include the same prior and the PPO ratio stays consistent (the prior is part of the
 * policy parameterization, recomputed deterministically from the same valid set). Use
 * this for the learned agents (UnknownAgent / FrozenAgent) — NOT for `RandomBot`, which
 * reads the raw 0/MASK_NEG validity mask as a boolean ("allowed iff === 0").
 */
export function computeActionMaskWithPrior(eid: number, di = GameDI): Float32Array {
  const mask = computeActionMask(eid, di);

  for (const group of PRIOR_GROUPS) {
    let validCount = 0;
    for (let i = group.start; i < group.start + group.count; i++) {
      if (mask[i] === 0) validCount++;
    }
    if (validCount === 0) continue; // whole group forbidden — leave it at MASK_NEG

    const prior = Math.log(group.share / validCount);
    for (let i = group.start; i < group.start + group.count; i++) {
      if (mask[i] === 0) mask[i] = prior;
    }
  }

  return mask;
}
