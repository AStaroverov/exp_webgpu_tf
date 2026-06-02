/**
 * applyActionToGame — decode sampled categorical action indices into exactly one
 * `enqueueAction` call (the decision seam). Heads irrelevant to the chosen kind
 * are ignored. Invalid choices (no passable neighbour / no enemy) become no-ops;
 * the reward's time penalty discourages them (PLAN §5.1, v1 no-op fallback).
 *
 * actions = [kind, moveDir, fireTarget] (Float32Array from batchAct).
 */

import { query } from 'bitecs';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { MapDI } from '../../../unknown/src/Game/DI/MapDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { enqueueAction } from '../../../unknown/src/Game/ECS/Actions/ActionSchedule.ts';
import { ActionKind, TargetKind } from '../../../unknown/src/Game/ECS/Actions/ActionTypes.ts';
import { HOLD_DURATION_MS, K_ENEMY, MOVE_SPEED, PolicyActionKind } from '../consts.ts';

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
        const neighbors = grid.neighbors({ q: here.q, r: here.r });
        const dest = neighbors[moveDir];
        if (!dest || !grid.isPassable(dest.q, dest.r)) return; // invalid → no-op
        enqueueAction(eid, {
            kind: ActionKind.MoveStep,
            target: { kind: TargetKind.Hex, q: dest.q, r: dest.r },
            params: { speed: MOVE_SPEED },
        });
        return;
    }

    if (kind === PolicyActionKind.Fire) {
        const target = nearestEnemies(eid, px, py, world)[fireTarget];
        if (!target) return; // no enemy in that slot → no-op
        enqueueAction(eid, {
            kind: ActionKind.Fire,
            target: { kind: TargetKind.Hex, q: target.q, r: target.r },
        });
        return;
    }
}

/** Up to K_ENEMY enemy hexes, nearest first — the fire-head slot list. */
function nearestEnemies(
    selfEid: number,
    px: number,
    py: number,
    world: (typeof GameDI)['world'],
): Array<{ q: number; r: number }> {
    const { Tank, TeamRef, RigidBodyState } = getGameComponents(world);
    const grid = MapDI.grid;
    const myTeam = TeamRef.id[selfEid];

    const enemies: Array<{ q: number; r: number; d: number }> = [];
    const tanks = query(world, [Tank]);
    for (let i = 0; i < tanks.length; i++) {
        const other = tanks[i];
        if (other === selfEid || TeamRef.id[other] === myTeam) continue;
        const ox = RigidBodyState.position.get(other, 0);
        const oy = RigidBodyState.position.get(other, 1);
        const hex = grid.worldToHex(ox, oy);
        if (!hex) continue;
        enemies.push({ q: hex.q, r: hex.r, d: (ox - px) ** 2 + (oy - py) ** 2 });
    }

    enemies.sort((a, b) => a.d - b.d);
    return enemies.slice(0, K_ENEMY);
}
