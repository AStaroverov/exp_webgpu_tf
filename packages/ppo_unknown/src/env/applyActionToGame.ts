/**
 * applyActionToGame — decode sampled categorical action indices into exactly one
 * `enqueueAction` call (the decision seam). Heads irrelevant to the chosen kind
 * are ignored. Invalid choices (no passable neighbour / empty fire cell) become
 * no-ops; the reward's time penalty discourages them (PLAN §5.1, v1 no-op
 * fallback). With action masking on (`computeActionMask`), these no-ops should be
 * unreachable in practice, but the all-masked guard means an empty-board step can
 * still arrive here — so the defensive fallbacks stay.
 *
 * actions = [kind, moveDir, fireTarget] (Float32Array from batchAct), where
 * `fireTarget` is a NEIGHBOUR DIRECTION 0..5 (same layout as `moveDir`).
 */

import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { MapDI } from '../../../unknown/src/Game/DI/MapDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { enqueueAction } from '../../../unknown/src/Game/ECS/Actions/ActionSchedule.ts';
import { ActionKind, TargetKind } from '../../../unknown/src/Game/ECS/Actions/ActionTypes.ts';
import type { HexGrid } from '../../../unknown/src/Game/Map/HexGrid.ts';
import { HOLD_DURATION_MS, MOVE_SPEED, PolicyActionKind } from '../consts.ts';

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
    q: number,
    r: number,
    moveDir: number,
): { q: number; r: number } | undefined {
    const dest = grid.neighborAt({ q, r }, moveDir);
    if (!dest || !grid.isPassable(dest.q, dest.r)) return undefined;
    return { q: dest.q, r: dest.r };
}

export function applyActionToGame(eid: number, actions: Float32Array, { world } = GameDI): void {
    const grid = MapDI.grid;
    if (!grid) return;

    const { RigidBodyState } = getGameComponents(world);
    const px = RigidBodyState.position.get(eid, 0);
    const py = RigidBodyState.position.get(eid, 1);
    const here = grid.worldToHex(px, py);
    if (!here) return;

    const kind = actions[0] | 0;
    const moveDir = actions[1] | 0;
    const fireTarget = actions[2] | 0;

    if (kind === PolicyActionKind.Hold) {
        enqueueAction(eid, { kind: ActionKind.Hold, params: { duration: HOLD_DURATION_MS } });
        return;
    }

    if (kind === PolicyActionKind.MoveStep) {
        const dest = moveDestination(grid, here.q, here.r, moveDir);
        if (!dest) return; // invalid → no-op
        enqueueAction(eid, {
            kind: ActionKind.MoveStep,
            target: { kind: TargetKind.Hex, q: dest.q, r: dest.r },
            params: { speed: MOVE_SPEED },
        });
        return;
    }

    if (kind === PolicyActionKind.Fire) {
        // fireTarget is a neighbour direction 0..5 (same layout as moveDir): fire at
        // the in-grid neighbour hex in that FIXED direction. neighborAt keeps the slot
        // direction-stable (NOT the compacted neighbors), matching computeActionMask.
        const target = grid.neighborAt(here, fireTarget);
        if (!target) return; // off-grid direction → no-op (defensive)
        enqueueAction(eid, {
            kind: ActionKind.Fire,
            target: { kind: TargetKind.Hex, q: target.q, r: target.r },
        });
        return;
    }
}
