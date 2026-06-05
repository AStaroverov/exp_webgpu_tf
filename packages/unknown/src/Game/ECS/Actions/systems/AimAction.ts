/**
 * Aim executor — rotates the owner tank's turret to point at the action's
 * target (entity / hex / point). Each tick it computes the desired world angle
 * from the turret toward the target and nudges `TurretController.rotation` (the
 * direction modifier the physics turret-rotation system integrates) toward it,
 * with proportional slow-down near the goal to avoid overshoot. The action
 * finishes once the heading error is within `tolerance`.
 *
 * Acts only on slot-0 fronts of its kind; owners run concurrently.
 */

import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { MapDI } from '../../../DI/MapDI.ts';
import { normalizeAngle } from '../../../../../../../lib/math.ts';
import { getGameComponents } from '../../createGameWorld.ts';
import { ActionDescriptor, encodeTarget } from '../ActionDescriptor.ts';
import { ActionKind, ActionStatus, ActionWorldTargetSpec, TargetKind } from '../ActionTypes.ts';
import { AimParamOffset } from '../ActionSlot.ts';

/** Heading-error band (rad) below which we steer proportionally instead of full speed. */
const SLOW_BAND = 0.3;

/** Strict params for an Aim action. `tolerance` = aim accuracy (radians). */
export type AimParamsSpec = { tolerance: number };

/** Enqueue spec for an Aim action. */
export type AimActionSpec = {
    kind: ActionKind.Aim;
    target: ActionWorldTargetSpec;
    params: AimParamsSpec;
};

export const AimActionDescriptor: ActionDescriptor<AimActionSpec> = {
    kind: ActionKind.Aim,
    encode(eid, slot, spec) {
        const { ActionsQueue } = getGameComponents(GameDI.world);
        ActionsQueue.setKind(eid, slot, ActionKind.Aim);
        encodeTarget(ActionsQueue, eid, slot, spec.target);
        ActionsQueue.setParam(eid, slot, AimParamOffset.tolerance, spec.params.tolerance);
    },
    createSystem: () => createAimActionSystem(),
};

export function createAimActionSystem({ world } = GameDI) {
    const { ActionsQueue, Vehicle, Tank, TurretController, RigidBodyState } = getGameComponents(world);

    return function updateAim(_delta: number) {
        const eids = query(world, [ActionsQueue, Vehicle, RigidBodyState]);
        for (const ownerEid of eids) {
            if (ActionsQueue.count[ownerEid] === 0) continue;
            if (ActionsQueue.getKind(ownerEid, 0) !== ActionKind.Aim) continue;
            if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Finished) continue;

            const turretEid = Tank.turretEId[ownerEid];

            // No turret to aim → nothing we can do; finish.
            if (!turretEid) {
                ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
                continue;
            }

            // Resolve the target world point.
            let tx: number | null = null;
            let ty: number | null = null;
            switch (ActionsQueue.getTargetKind(ownerEid, 0)) {
                case TargetKind.Entity: {
                    const targetEid = ActionsQueue.getTargetVal(ownerEid, 0, 0);
                    tx = RigidBodyState.position.get(targetEid, 0);
                    ty = RigidBodyState.position.get(targetEid, 1);
                    break;
                }
                case TargetKind.Hex: {
                    const c = MapDI.grid.hexToWorld({
                        q: ActionsQueue.getTargetVal(ownerEid, 0, 0),
                        r: ActionsQueue.getTargetVal(ownerEid, 0, 1),
                    });
                    if (c) {
                        tx = c.x;
                        ty = c.y;
                    }
                    break;
                }
                case TargetKind.Point: {
                    tx = ActionsQueue.getTargetVal(ownerEid, 0, 0);
                    ty = ActionsQueue.getTargetVal(ownerEid, 0, 1);
                    break;
                }
            }

            if (tx === null || ty === null) {
                // Unresolvable target — fail the action.
                TurretController.setRotation$(turretEid, 0);
                ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
                continue;
            }

            if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Idle) {
                ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Running);
                // Tank is stationary while aiming, so the next decision's observation
                // is already valid — open the slot immediately (§4).
                ActionsQueue.scheduleRequestNext(ownerEid, 0);
            }

            const turretX = RigidBodyState.position.get(turretEid, 0);
            const turretY = RigidBodyState.position.get(turretEid, 1);
            const desired = Math.atan2(ty - turretY, tx - turretX);
            const err = normalizeAngle(desired - RigidBodyState.rotation[turretEid]);

            const tolerance = ActionsQueue.getParam(ownerEid, 0, AimParamOffset.tolerance) || 0.05;
            if (Math.abs(err) <= tolerance) {
                TurretController.setRotation$(turretEid, 0);
                ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
                continue;
            }

            // Proportional within SLOW_BAND, full speed outside it.
            const dir = Math.abs(err) >= SLOW_BAND ? Math.sign(err) : err / SLOW_BAND;
            TurretController.setRotation$(turretEid, dir);
        }
    };
}
