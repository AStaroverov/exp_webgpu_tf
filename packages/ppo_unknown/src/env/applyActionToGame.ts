/**
 * applyActionToGame — decode the sampled flat action index into exactly one
 * `enqueueAction` call (the decision seam). Invalid choices (no passable
 * neighbour / off-grid fire direction) become no-ops; the reward's time penalty
 * discourages them (PLAN §5.1, v1 no-op fallback). With action masking on
 * (`computeActionMask`) these no-ops should be unreachable in practice, but the
 * defensive fallbacks stay.
 *
 * actions = [index] (Float32Array from batchAct) — an index into the flat action
 * list `Hold | move dir 0..5 | fire target 0..35` (see consts.ts layout).
 */

import { hasComponent } from "bitecs";
import { GameDI } from "../../../unknown/src/Game/DI/GameDI.ts";
import { MapDI } from "../../../unknown/src/Game/DI/MapDI.ts";
import { getGameComponents } from "../../../unknown/src/Game/ECS/createGameWorld.ts";
import { enqueueAction } from "../../../unknown/src/Game/ECS/Actions/ActionSchedule.ts";
import { ActionKind, TargetKind } from "../../../unknown/src/Game/ECS/Actions/ActionTypes.ts";
import type { HexGrid } from "../../../unknown/src/Game/Map/HexGrid.ts";
import {
  HOLD_ACTION,
  HOLD_DURATION_MS,
  FIRE_ACTION_OFFSET,
  FIRE_TARGET_OFFSETS,
  MOVE_ACTION_OFFSET,
  MOVE_SPEED,
} from "../consts.ts";

/**
 * Shared move-passability predicate (single source of truth with
 * `computeActionMask`): the in-grid neighbour of `(q, r)` in the FIXED direction
 * slot `moveDir` (0..5, stable `POINTY_DIRECTIONS` order) that a unit may step
 * into, or `undefined` if that direction is off-grid / blocked. Uses the
 * direction-stable `neighborAt` (NOT the compacted `neighbors`) so action slot
 * `moveDir` means the same physical direction at every board position.
 */
export function moveDestination(
  grid: HexGrid,
  eid: number,
  q: number,
  r: number,
  moveDir: number,
): { q: number; r: number } | undefined {
  const dest = grid.neighborAt({ q, r }, moveDir);
  // `isPassableFor`: the unit's own buffer-ring reservation must not block it.
  if (!dest || !grid.isPassableFor(dest.q, dest.r, eid)) return undefined;
  return { q: dest.q, r: dest.r };
}

export function applyActionToGame(eid: number, actions: Float32Array, { world } = GameDI): void {
  const grid = MapDI.grid;
  if (!grid) return;

  const { RigidBodyState, Tank, StreamFirearms } = getGameComponents(world);
  const px = RigidBodyState.position.get(eid, 0);
  const py = RigidBodyState.position.get(eid, 1);
  const here = grid.worldToHex(px, py);
  if (!here) return;

  const action = actions[0] | 0;

  if (action === HOLD_ACTION) {
    enqueueAction(eid, { kind: ActionKind.Hold, params: { duration: HOLD_DURATION_MS } });
    return;
  }

  if (action < FIRE_ACTION_OFFSET) {
    const dest = moveDestination(grid, eid, here.q, here.r, action - MOVE_ACTION_OFFSET);
    if (!dest) return; // invalid → no-op
    enqueueAction(eid, {
      kind: ActionKind.MoveStep,
      target: { kind: TargetKind.Hex, q: dest.q, r: dest.r },
      params: { speed: MOVE_SPEED },
    });
    return;
  }

  const offset = FIRE_TARGET_OFFSETS[action - FIRE_ACTION_OFFSET];
  if (!offset) return;
  const target = { q: here.q + offset[0], r: here.r + offset[1] };
  if (!grid.has(target)) return;
  if (hasComponent(world, Tank.turretEId.get(eid), StreamFirearms)) {
    enqueueAction(eid, {
      kind: ActionKind.FireStream,
      target: { kind: TargetKind.Hex, q: target.q, r: target.r },
    });
    return;
  } else {
    enqueueAction(eid, {
      kind: ActionKind.Fire,
      target: { kind: TargetKind.Hex, q: target.q, r: target.r },
    });
    return;
  }
}
