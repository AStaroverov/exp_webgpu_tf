/**
 * Fire executor — aims the owner tank's turret at a target, then fires exactly one
 * round. A self-contained "shoot at this cell" action (target like Aim).
 *
 * Three in-slot phases (param `phase`):
 *   1. AIMING     — rotate the turret toward the target (proportional slow-down near
 *                   the goal); advance once heading error ≤ TOLERANCE. The hex target
 *                   is a *direction*: if a unit sits on a hex along the trajectory,
 *                   the aim refines onto that unit's actual body center.
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
import { OccupantKind } from '../../../Map/HexGrid.ts';
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

    const sharedAimPoint = { x: 0, y: 0 };
    function tick(_delta: number) {
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
                ActionsQueue.setParam(ownerEid, 0, FireParamOffset.phase, FIRE_PHASE_AIMING);
                ActionsQueue.scheduleRequestNext(ownerEid, 0);
            }

            const phase = ActionsQueue.getParam(ownerEid, 0, FireParamOffset.phase);

            if (phase === FIRE_PHASE_AIMING) {
                const targetQ = ActionsQueue.getTargetVal(ownerEid, 0, 0);
                const targetR = ActionsQueue.getTargetVal(ownerEid, 0, 1);
                // Resolve the target hex's world center.
                const targetCenter = MapDI.grid.hexToWorld(targetQ, targetR);

                // Off-map target → can't aim; abort.
                if (!targetCenter) {
                    TurretController.setRotation$(turretEid, 0);
                    ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
                    continue;
                }

                // Re-resolved every tick so the aim tracks a moving unit.
                const aimPoint = resolveAimPoint(ownerEid, targetQ, targetR, sharedAimPoint) ?? targetCenter;

                const turretX = RigidBodyState.position.get(turretEid, 0);
                const turretY = RigidBodyState.position.get(turretEid, 1);
                const desired = Math.atan2(aimPoint.y - turretY, aimPoint.x - turretX);
                const err = normalizeAngle(desired - RigidBodyState.rotation[turretEid]);

                if (Math.abs(err) <= TOLERANCE) {
                    TurretController.setRotation$(turretEid, 0);
                    // On target → wait for the weapon, then fire a round.
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

    const sharedTarget = { q: 0, r: 0 };
    function resolveAimPoint(
        ownerEid: number,
        targetQ: number,
        targetR: number,
        out?: { x: number; y: number }
    ): { x: number; y: number } | null {
        const grid = MapDI.grid;
        const here = grid.worldToHex(
            RigidBodyState.position.get(ownerEid, 0),
            RigidBodyState.position.get(ownerEid, 1),
        );
        if (!here) return null;

        // The target is a neighbour hex, so the delta is one axial step — adding it
        // repeatedly walks the straight hex ray in that direction.
        const dq = targetQ - here.q;
        const dr = targetR - here.r;
        if (dq === 0 && dr === 0) return null;

        sharedTarget.q = here.q;
        sharedTarget.r = here.r;
        while (true) {
            sharedTarget.q += dq;
            sharedTarget.r += dr;
            if (!grid.has(sharedTarget)) return null;
            const occupant = grid.getOccupant(sharedTarget.q, sharedTarget.r);
            if (!occupant || occupant.eid === ownerEid) continue;
            if (occupant.kind === OccupantKind.Unit) {
                out = out ?? { x: 0, y: 0 };
                out.x = RigidBodyState.position.get(occupant.eid, 0);
                out.y = RigidBodyState.position.get(occupant.eid, 1);
                return out;
            }
            if (occupant.kind === OccupantKind.Obstacle) return null;
            // Reserved → keep scanning past it.
        }
    }

    return tick;
}

