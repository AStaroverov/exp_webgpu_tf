/**
 * MoveStep executor — atomic one-hop move with no A* pathfinding and no waypoint
 * list: the target hex is a single neighbour cell and the whole "path" is exactly
 * its center (`grid.hexToWorld(target)`). The owner's VehicleController steers
 * straight at that one center and finishes on arrival.
 *
 * This executor does NOT mutate grid occupancy. The grid's Unit/Reserved layer is
 * rebuilt every tick from vehicle positions + velocity by `createGridOccupancySystem`
 * (which runs before actions), so MoveStep only *reads* it: each tick, if the
 * target cell is held by ANOTHER unit, it just stops and finishes — never drives
 * into a cell that isn't its own. Stopping → owner idle → a fresh slot opens.
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
import { MoveStepParamOffset } from '../ActionSlot.ts';

/** Distance (world units) within which the target center counts as reached. */
const ARRIVE_EPSILON = 12;
/**
 * Proximity band (world units, > ARRIVE_EPSILON) to the destination center at
 * which we open the slot, so the next decision is computed from a near-final
 * position rather than a stale start one (§4).
 */
const REQUEST_NEXT_DIST = 18;
/** Heading error (rad) below which we drive straight instead of pivoting. */
const HEADING_DEADZONE = 0.15;

/** Strict params for a MoveStep action. */
export type MoveStepParamsSpec = { speed: number };

/** Enqueue spec for a MoveStep action. */
export type MoveStepActionSpec = {
    kind: ActionKind.MoveStep;
    target: ActionHexTargetSpec;
    params: MoveStepParamsSpec;
};

export const MoveStepActionDescriptor: ActionDescriptor<MoveStepActionSpec> = {
    kind: ActionKind.MoveStep,
    encode(eid, slot, spec) {
        const { ActionsQueue } = getGameComponents(GameDI.world);
        ActionsQueue.setKind(eid, slot, ActionKind.MoveStep);
        encodeTarget(ActionsQueue, eid, slot, spec.target);
        ActionsQueue.setParam(eid, slot, MoveStepParamOffset.speed, spec.params.speed);
    },
    createSystem: () => createMoveStepActionSystem(),
};

export function createMoveStepActionSystem({ world } = GameDI) {
    const { ActionsQueue, Vehicle, VehicleController, RigidBodyState } = getGameComponents(world);

    const stop = (ownerEid: number) => {
        VehicleController.setMove$(ownerEid, 0);
        VehicleController.setRotate$(ownerEid, 0);
    };

    return function updateMoveStep(_delta: number) {
        const grid = MapDI.grid;
        const eids = query(world, [ActionsQueue, Vehicle, RigidBodyState]);
        for (const ownerEid of eids) {
            if (ActionsQueue.count[ownerEid] === 0) continue;
            if (ActionsQueue.getKind(ownerEid, 0) !== ActionKind.MoveStep) continue;
            if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Finished) continue;

            const targetQ = ActionsQueue.getTargetVal(ownerEid, 0, 0);
            const targetR = ActionsQueue.getTargetVal(ownerEid, 0, 1);
            const targetCenter = grid.hexToWorld({ q: targetQ, r: targetR });

            const px = RigidBodyState.position.get(ownerEid, 0);
            const py = RigidBodyState.position.get(ownerEid, 1);

            if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Idle) {
                ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Running);
            }

            // Off-map target → nowhere to go.
            if (!targetCenter) {
                stop(ownerEid);
                ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
                continue;
            }

            // Self-check: occupancy is owned by createGridOccupancySystem (rebuilt each
            // tick from positions + velocity). If the target cell is held by ANOTHER
            // unit, just stop — never drive into a cell that isn't ours. A null cell or
            // our own mark is fine (while rotating we have no velocity, so the cell
            // ahead may be unmarked yet).
            const occ = grid.getOccupant(targetQ, targetR);
            if (occ !== null && occ.eid !== ownerEid) {
                stop(ownerEid);
                ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
                continue;
            }

            // Arrival.
            if (Math.hypot(targetCenter.x - px, targetCenter.y - py) <= ARRIVE_EPSILON) {
                stop(ownerEid);
                ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
                continue;
            }

            // Open the slot when close to the destination (idempotent; front only).
            if (
                !ActionsQueue.shouldRequestNext(ownerEid, 0) &&
                Math.hypot(targetCenter.x - px, targetCenter.y - py) <= REQUEST_NEXT_DIST
            ) {
                ActionsQueue.scheduleRequestNext(ownerEid, 0);
            }

            // Steer toward the single target center.
            const desired = Math.atan2(targetCenter.y - py, targetCenter.x - px);
            const heading = RigidBodyState.rotation[ownerEid];
            const err = normalizeAngle(desired - heading);

            VehicleController.setRotate$(ownerEid, Math.abs(err) < 1e-3 ? 0 : Math.sign(err));
            VehicleController.setMove$(ownerEid, Math.abs(err) < HEADING_DEADZONE ? 1 : 0.3);
        }
    };
}
