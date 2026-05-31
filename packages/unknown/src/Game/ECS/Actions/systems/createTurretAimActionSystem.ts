/**
 * TurretAim executor — rotates the owner tank's turret to point at the action's
 * target (entity / hex / point). Each tick it computes the desired world angle
 * from the turret toward the target and nudges `TurretController.rotation` (the
 * direction modifier the physics turret-rotation system integrates) toward it,
 * with proportional slow-down near the goal to avoid overshoot. The action
 * finishes once the heading error is within `tolerance`.
 *
 * Only the global top action of this kind is executed (chess-like sequencing).
 */

import { MapDI } from '../../../DI/MapDI.ts';
import { normalizeAngle } from '../../../../../../../lib/math.ts';
import { addEntity, World } from 'bitecs';
import { getPhysicsWorldComponents, PhysicsWorld } from '../../createPhysicsWorld.ts';
import { ActionDescriptor, applyTarget } from '../ActionDescriptor.ts';
import { getTopAction } from '../ActionScheduleDI.ts';
import { getActionComponents } from '../createActionWorld.ts';
import { ActionKind, ActionStatus, ActionWorldTargetSpec, TargetKind } from '../ActionTypes.ts';

/** Heading-error band (rad) below which we steer proportionally instead of full speed. */
const SLOW_BAND = 0.3;

/** Strict params for a TurretAim action. `tolerance` = aim accuracy (radians). */
export type TurretAimParamsSpec = { tolerance: number };

/** Enqueue spec for a TurretAim action. */
export type TurretAimActionSpec = {
    kind: ActionKind.TurretAim;
    target: ActionWorldTargetSpec;
    params: TurretAimParamsSpec;
};

export const TurretAimActionDescriptor: ActionDescriptor<TurretAimActionSpec> = {
    kind: ActionKind.TurretAim,
    createSystem: (actionWorld, gameWorld) => createTurretAimActionSystem(actionWorld, gameWorld),
    createAction(world, ownerEid, spec, seq) {
        const { Action, TurretAimParams } = getActionComponents(world);
        const eid = addEntity(world);
        Action.addComponent(world, eid, ActionKind.TurretAim, ownerEid, seq);
        applyTarget(world, eid, spec.target);
        TurretAimParams.addComponent(world, eid, spec.params.tolerance);
        return eid;
    },
};

export function createTurretAimActionSystem(
    actionWorld: World,
    gameWorld: PhysicsWorld,
) {
    const { Action, ActionTarget, TurretAimParams } = getActionComponents(actionWorld);
    const { Tank, TurretController, RigidBodyState } = getPhysicsWorldComponents(gameWorld);

    return function updateTurretAim(_delta: number) {
        const top = getTopAction();
        if (top === null) return;
        if (Action.kind[top] !== ActionKind.TurretAim) return;
        if (Action.status[top] === ActionStatus.Finished) return;

        const ownerEid = Action.ownerEid[top];
        const turretEid = Tank.turretEId[ownerEid];

        // No turret to aim → nothing we can do; finish.
        if (!turretEid) {
            Action.setFinished$(top);
            return;
        }

        // Resolve the target world point.
        let tx: number | null = null;
        let ty: number | null = null;
        switch (ActionTarget.kind[top]) {
            case TargetKind.Entity: {
                const targetEid = ActionTarget.values.get(top, 0);
                tx = RigidBodyState.position.get(targetEid, 0);
                ty = RigidBodyState.position.get(targetEid, 1);
                break;
            }
            case TargetKind.Hex: {
                const c = MapDI.grid.hexToWorld({
                    q: ActionTarget.values.get(top, 0),
                    r: ActionTarget.values.get(top, 1),
                });
                if (c) {
                    tx = c.x;
                    ty = c.y;
                }
                break;
            }
            case TargetKind.Point: {
                tx = ActionTarget.values.get(top, 0);
                ty = ActionTarget.values.get(top, 1);
                break;
            }
        }

        if (tx === null || ty === null) {
            // Unresolvable target — fail the action.
            TurretController.setRotation$(turretEid, 0);
            Action.setFinished$(top);
            return;
        }

        if (Action.status[top] === ActionStatus.Idle) {
            Action.setRunning$(top);
        }

        const turretX = RigidBodyState.position.get(turretEid, 0);
        const turretY = RigidBodyState.position.get(turretEid, 1);
        const desired = Math.atan2(ty - turretY, tx - turretX);
        const err = normalizeAngle(desired - RigidBodyState.rotation[turretEid]);

        const tolerance = TurretAimParams.tolerance[top] || 0.05;
        if (Math.abs(err) <= tolerance) {
            TurretController.setRotation$(turretEid, 0);
            Action.setFinished$(top);
            return;
        }

        // Proportional within SLOW_BAND, full speed outside it.
        const dir = Math.abs(err) >= SLOW_BAND ? Math.sign(err) : err / SLOW_BAND;
        TurretController.setRotation$(turretEid, dir);
    };
}
