/**
 * Fire executor — fires the owner tank's weapon `shots` times, then finishes.
 *
 * Firing is indirect: the bullet spawner (`createSpawnerBulletsSystem`, run later
 * in the same tick during spawnFrame) reads `TurretController.shoot` and spawns a
 * round only when the turret is NOT reloading, then starts the reload timer. So a
 * single round is fired by: waiting until the turret is ready (`!isReloading`),
 * raising the shoot flag, and detecting the shot on the next tick via the freshly
 * started reload. We repeat that for each requested shot.
 *
 * Per-action state is kept in a `Map` keyed by the action eid, rebuilt whenever a
 * fresh (Idle) action is picked up so reused entity ids never inherit stale state.
 * Only the global top action of this kind is executed (chess-like sequencing).
 */

import { addEntity, World } from 'bitecs';
import { getPhysicsWorldComponents, PhysicsWorld } from '../../createPhysicsWorld.ts';
import { ActionDescriptor, applyTarget } from '../ActionDescriptor.ts';
import { getTopAction } from '../ActionScheduleDI.ts';
import { getActionComponents } from '../createActionWorld.ts';
import { ActionKind, ActionStatus } from '../ActionTypes.ts';

type FirePlan = {
    /** 'waitReady' = waiting for the turret to finish reloading before firing the next round. */
    /** 'firing' = shoot flag raised; waiting for the reload that confirms the round was fired. */
    phase: 'waitReady' | 'firing';
    remaining: number;
};

/** Strict params for a Fire action. `shots` = number of rounds to fire. */
export type FireParamsSpec = { shots: number };

/** Enqueue spec for a Fire action (no target — fires where the turret points). */
export type FireActionSpec = {
    kind: ActionKind.Fire;
    params: FireParamsSpec;
};

export const FireActionDescriptor: ActionDescriptor<FireActionSpec> = {
    kind: ActionKind.Fire,
    createSystem: (actionWorld, gameWorld) => createFireActionSystem(actionWorld, gameWorld),
    createAction(world, ownerEid, spec, seq) {
        const { Action, FireParams } = getActionComponents(world);
        const eid = addEntity(world);
        Action.addComponent(world, eid, ActionKind.Fire, ownerEid, seq);
        applyTarget(world, eid);
        FireParams.addComponent(world, eid, spec.params.shots);
        return eid;
    },
};

export function createFireActionSystem(
    actionWorld: World,
    gameWorld: PhysicsWorld,
) {
    const { Action, FireParams } = getActionComponents(actionWorld);
    const { Tank, TurretController, Firearms } = getPhysicsWorldComponents(gameWorld);

    const plans = new Map<number, FirePlan>();

    return function updateFire(_delta: number) {
        const top = getTopAction();
        if (top === null) return;
        if (Action.kind[top] !== ActionKind.Fire) return;
        if (Action.status[top] === ActionStatus.Finished) return;

        const ownerEid = Action.ownerEid[top];
        const turretEid = Tank.turretEId[ownerEid];

        // No turret to fire from → finish immediately.
        if (!turretEid) {
            plans.delete(top);
            Action.setFinished$(top);
            return;
        }

        if (Action.status[top] === ActionStatus.Idle) {
            Action.setRunning$(top);
            plans.set(top, { phase: 'waitReady', remaining: Math.max(1, FireParams.shots[top] || 1) });
        }

        const plan = plans.get(top);
        if (!plan) {
            TurretController.setShooting$(turretEid, 0);
            Action.setFinished$(top);
            return;
        }

        if (plan.phase === 'waitReady') {
            // Wait until the weapon is ready, then raise the shoot flag. The spawner
            // (run later this tick) fires the round and starts the reload.
            if (Firearms.isReloading(turretEid)) return;
            TurretController.setShooting$(turretEid, 1);
            plan.phase = 'firing';
            return;
        }

        // phase === 'firing': the reload starting confirms the round was fired.
        if (Firearms.isReloading(turretEid)) {
            TurretController.setShooting$(turretEid, 0);
            plan.remaining--;
            if (plan.remaining <= 0) {
                plans.delete(top);
                Action.setFinished$(top);
            } else {
                plan.phase = 'waitReady';
            }
        }
    };
}
