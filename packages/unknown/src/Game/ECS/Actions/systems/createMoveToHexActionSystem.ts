/**
 * MoveToHex executor — drives the owner's VehicleController along an A* path to
 * the target hex, passing through each cell CENTER, and updates grid occupancy.
 *
 * The path is computed ONCE when the action starts (status Idle → Running) and
 * stored as a list of world-space waypoints (cell centers). Each tick the vehicle
 * steers toward the current waypoint; when it reaches that center it advances to
 * the next one. Re-deriving the aim from the current cell every tick is what made
 * the vehicle cut corners (aim jumping past the turn) and spin in place (aiming at
 * the cell center it had already passed, i.e. backwards) — a fixed forward
 * waypoint list avoids both. Only the global top action of this kind is executed.
 */

import { addEntity, World } from 'bitecs';
import { MapDI } from '../../../DI/MapDI.ts';
import { MapWorldId } from '../../../Map/HexGrid.ts';
import { findPath } from '../../../Map/findPath.ts';
import { normalizeAngle } from '../../../../../../../lib/math.ts';
import { getPhysicsWorldComponents, PhysicsWorld } from '../../createPhysicsWorld.ts';
import { getBrainWorldComponents } from '../../createBrainWorld.ts';
import { getNodeByPhysics } from '../../refs.ts';
import { Worlds } from '../../../DI/Worlds.ts';
import { ActionDescriptor, applyTarget } from '../ActionDescriptor.ts';
import { getTopAction } from '../ActionScheduleDI.ts';
import { getActionComponents } from '../createActionWorld.ts';
import { ActionHexTargetSpec, ActionKind, ActionStatus } from '../ActionTypes.ts';

/** Distance (world units) within which a waypoint center counts as reached. */
const ARRIVE_EPSILON = 12;
/** Heading error (rad) below which we drive straight instead of pivoting. */
const HEADING_DEADZONE = 0.15;

type MovePlan = {
    centers: Array<{ x: number; y: number }>;
    idx: number;
    fromQ: number;
    fromR: number;
};

/** Strict params for a MoveToHex action. */
export type MoveToHexParamsSpec = { speed: number };

/** Enqueue spec for a MoveToHex action. */
export type MoveToHexActionSpec = {
    kind: ActionKind.MoveToHex;
    target: ActionHexTargetSpec;
    params: MoveToHexParamsSpec;
};

export const MoveToHexActionDescriptor: ActionDescriptor<MoveToHexActionSpec> = {
    kind: ActionKind.MoveToHex,
    createAction(world, ownerEid, spec, seq) {
        const { Action, MoveToHexParams } = getActionComponents(world);
        const eid = addEntity(world);
        Action.addComponent(world, eid, ActionKind.MoveToHex, ownerEid, seq);
        applyTarget(world, eid, spec.target);
        MoveToHexParams.addComponent(world, eid, spec.params.speed);
        return eid;
    },
    createSystem: (actionWorld, gameWorld) => createMoveToHexActionSystem(actionWorld, gameWorld),
};

export function createMoveToHexActionSystem(
    actionWorld: World,
    gameWorld: PhysicsWorld,
) {
    const { Action, ActionTarget } = getActionComponents(actionWorld);
    const { RigidBodyState } = getPhysicsWorldComponents(gameWorld);
    const { VehicleController } = getBrainWorldComponents(Worlds.brainWorld);

    // Per-action movement plans. Keyed by the action entity id; rebuilt whenever a
    // fresh (Idle) action is picked up, so reused entity ids never inherit a stale
    // plan.
    const plans = new Map<number, MovePlan>();

    // ownerEid is the hull ATOM; the controller lives on its hull node (hull-brain).
    const stop = (ownerEid: number) => {
        const brain = getNodeByPhysics(ownerEid);
        VehicleController.setMove$(brain, 0);
        VehicleController.setRotate$(brain, 0);
    };

    return function updateMoveToHex(_delta: number) {
        const top = getTopAction();
        if (top === null) return;
        if (Action.kind[top] !== ActionKind.MoveToHex) return;
        if (Action.status[top] === ActionStatus.Finished) return;

        const grid = MapDI.grid;
        const ownerEid = Action.ownerEid[top];
        const goalQ = ActionTarget.values.get(top, 0);
        const goalR = ActionTarget.values.get(top, 1);

        const px = RigidBodyState.position.get(ownerEid, 0);
        const py = RigidBodyState.position.get(ownerEid, 1);

        const goalCenter = grid.hexToWorld({ q: goalQ, r: goalR });
        if (!goalCenter) {
            // Target hex does not exist — fail the action.
            stop(ownerEid);
            plans.delete(top);
            Action.setFinished$(top);
            return;
        }

        // Build the waypoint plan once, when the action first runs.
        if (Action.status[top] === ActionStatus.Idle) {
            Action.setRunning$(top);

            const startHex = grid.worldToHex(px, py);
            const centers: Array<{ x: number; y: number }> = [];
            if (startHex) {
                const path = findPath(
                    grid,
                    { q: startHex.q, r: startHex.r },
                    { q: goalQ, r: goalR },
                    {
                        // Allow stepping onto the goal even if it's marked occupied
                        // (e.g. reserved by us).
                        isBlocked: (q, r) =>
                            (q === goalQ && r === goalR) ? false : !grid.isPassable(q, r),
                    },
                );
                // Skip path[0] (the start cell — we're already there); the rest are
                // forward waypoints ending at the goal center.
                if (path) {
                    for (let i = 1; i < path.length; i++) {
                        const c = grid.hexToWorld(path[i]);
                        if (c) centers.push(c);
                    }
                }
            }
            // Fallback: head straight to the goal center if no path was produced.
            if (centers.length === 0) centers.push(goalCenter);

            plans.set(top, {
                centers,
                idx: 0,
                fromQ: startHex ? startHex.q : goalQ,
                fromR: startHex ? startHex.r : goalR,
            });
        }

        const plan = plans.get(top);
        if (!plan) {
            // Defensive: no plan (e.g. after a reset) — finish cleanly.
            stop(ownerEid);
            Action.setFinished$(top);
            return;
        }

        // Advance past any waypoints already reached (usually at most one/tick).
        while (plan.idx < plan.centers.length) {
            const wp = plan.centers[plan.idx];
            if (Math.hypot(wp.x - px, wp.y - py) <= ARRIVE_EPSILON) {
                plan.idx++;
            } else {
                break;
            }
        }

        // Reached the final waypoint (goal center) → arrive.
        if (plan.idx >= plan.centers.length) {
            stop(ownerEid);
            if (!(plan.fromQ === goalQ && plan.fromR === goalR)) {
                grid.vacate(plan.fromQ, plan.fromR);
            }
            grid.occupy(goalQ, goalR, ownerEid, MapWorldId.Game);
            plans.delete(top);
            Action.setFinished$(top);
            return;
        }

        // Steer toward the current waypoint.
        const aim = plan.centers[plan.idx];
        const desired = Math.atan2(aim.y - py, aim.x - px);
        const heading = RigidBodyState.rotation[ownerEid];
        const err = normalizeAngle(desired - heading);

        const ownerBrain = getNodeByPhysics(ownerEid);
        VehicleController.setRotate$(ownerBrain, Math.abs(err) < 1e-3 ? 0 : Math.sign(err));
        VehicleController.setMove$(ownerBrain, Math.abs(err) < HEADING_DEADZONE ? 1 : 0.3);
    };
}
