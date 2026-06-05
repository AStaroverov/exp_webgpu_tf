/**
 * Fire executor — aims the owner tank's turret at a target, then fires exactly one
 * round. A self-contained "shoot at this cell" action (target like Aim).
 *
 * Three in-slot phases (param `phase`):
 *   1. AIMING     — rotate the turret toward the target (proportional slow-down near
 *                   the goal); advance once heading error ≤ TOLERANCE.
 *   2. WAIT_READY — wait until the weapon is not reloading, then raise the shoot flag
 *                   (the bullet spawner, later this tick, fires the round + starts reload).
 *   3. FIRING     — the freshly started reload confirms the shot; lower the flag, finish.
 *
 * Acts only on slot-0 fronts of its kind; owners run concurrently.
 */

import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { MapDI } from '../../../DI/MapDI.ts';
import { normalizeAngle } from '../../../../../../../lib/math.ts';
import { getGameComponents } from '../../createGameWorld.ts';
import { ActionDescriptor, encodeTarget } from '../ActionDescriptor.ts';
import { ActionHexTargetSpec, ActionKind, ActionStatus } from '../ActionTypes.ts';
import {
    FireParamOffset,
    FIRE_PHASE_AIMING,
    FIRE_PHASE_WAIT_READY,
    FIRE_PHASE_FIRING,
} from '../ActionSlot.ts';

/** Heading-error band (rad) below which we steer proportionally instead of full speed. */
const SLOW_BAND = 0.3;
/** Heading error (rad) within which the turret counts as on-target and may fire. */
const TOLERANCE = 0.05;

/** Enqueue spec for a Fire action — aims at the target, then fires one round. */
export type FireActionSpec = {
    kind: ActionKind.Fire;
    target: ActionHexTargetSpec;
};

export const FireActionDescriptor: ActionDescriptor<FireActionSpec> = {
    kind: ActionKind.Fire,
    encode(eid, slot, spec) {
        const { ActionsQueue } = getGameComponents(GameDI.world);
        ActionsQueue.setKind(eid, slot, ActionKind.Fire);
        encodeTarget(ActionsQueue, eid, slot, spec.target);
        ActionsQueue.setParam(eid, slot, FireParamOffset.phase, FIRE_PHASE_AIMING);
    },
    createSystem: () => createFireActionSystem(),
};

export function createFireActionSystem({ world } = GameDI) {
    const { ActionsQueue, Vehicle, Tank, TurretController, Firearms, RigidBodyState } = getGameComponents(world);

    return function updateFire(_delta: number) {
        const eids = query(world, [ActionsQueue, Vehicle, RigidBodyState]);
        for (const ownerEid of eids) {
            if (ActionsQueue.count[ownerEid] === 0) continue;
            if (ActionsQueue.getKind(ownerEid, 0) !== ActionKind.Fire) continue;
            if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Finished) continue;

            const turretEid = Tank.turretEId[ownerEid];

            // No turret to fire from → finish immediately.
            if (!turretEid) {
                ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
                continue;
            }

            if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Idle) {
                ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Running);
                // Tank is stationary while aiming/firing, so the next decision's
                // observation is already valid — open the slot immediately (§4).
                ActionsQueue.scheduleRequestNext(ownerEid, 0);
                ActionsQueue.setParam(ownerEid, 0, FireParamOffset.phase, FIRE_PHASE_AIMING);
            }

            const phase = ActionsQueue.getParam(ownerEid, 0, FireParamOffset.phase);

            if (phase === FIRE_PHASE_AIMING) {
                // Resolve the target hex's world center.
                const targetCenter = MapDI.grid.hexToWorld({
                    q: ActionsQueue.getTargetVal(ownerEid, 0, 0),
                    r: ActionsQueue.getTargetVal(ownerEid, 0, 1),
                });

                // Off-map target → can't aim; abort.
                if (!targetCenter) {
                    TurretController.setRotation$(turretEid, 0);
                    ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
                    continue;
                }

                const turretX = RigidBodyState.position.get(turretEid, 0);
                const turretY = RigidBodyState.position.get(turretEid, 1);
                const desired = Math.atan2(targetCenter.y - turretY, targetCenter.x - turretX);
                const err = normalizeAngle(desired - RigidBodyState.rotation[turretEid]);

                if (Math.abs(err) <= TOLERANCE) {
                    // On target → stop rotating, move on to firing.
                    TurretController.setRotation$(turretEid, 0);
                    ActionsQueue.setParam(ownerEid, 0, FireParamOffset.phase, FIRE_PHASE_WAIT_READY);
                    continue;
                }

                // Proportional within SLOW_BAND, full speed outside it.
                const dir = Math.abs(err) >= SLOW_BAND ? Math.sign(err) : err / SLOW_BAND;
                TurretController.setRotation$(turretEid, dir);
                continue;
            }

            if (phase === FIRE_PHASE_WAIT_READY) {
                // Wait until the weapon is ready, then raise the shoot flag. The spawner
                // (run later this tick) fires the round and starts the reload.
                if (Firearms.isReloading(turretEid)) continue;
                TurretController.setShooting$(turretEid, 1);
                ActionsQueue.setParam(ownerEid, 0, FireParamOffset.phase, FIRE_PHASE_FIRING);
                continue;
            }

            // phase === FIRE_PHASE_FIRING: the reload starting confirms the round was fired.
            if (Firearms.isReloading(turretEid)) {
                TurretController.setShooting$(turretEid, 0);
                ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
            }
        }
    };
}
