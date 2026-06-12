/**
 * Aim executor — rotates the owner tank's turret to point at the action's
 * target hex. Each tick it computes the desired world angle
 * from the turret toward the target and nudges `TurretController.rotation` (the
 * direction modifier the physics turret-rotation system integrates) toward it,
 * with proportional slow-down near the goal to avoid overshoot. The action
 * finishes once the heading error is within `tolerance`.
 *
 * Acts only on slot-0 fronts of its kind; owners run concurrently.
 */

import { hasComponent, query } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { MapDI } from "../../../DI/MapDI.ts";
import { normalizeAngle } from "../../../../../../../lib/math.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import { ActionDescriptor, encodeTarget } from "../ActionDescriptor.ts";
import { ActionHexTargetSpec, ActionKind, ActionStatus } from "../ActionTypes.ts";
import { AimParamOffset } from "../ActionSlot.ts";

/** Heading-error band (rad) below which we steer proportionally instead of full speed. */
const SLOW_BAND = 0.3;

/** Strict params for an Aim action. `tolerance` = aim accuracy (radians). */
export type AimParamsSpec = { tolerance: number };

/** Enqueue spec for an Aim action. */
export type AimActionSpec = {
  kind: ActionKind.Aim;
  target: ActionHexTargetSpec;
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
  const {
    ActionsQueue,
    Vehicle,
    Tank,
    TurretController,
    VehicleController,
    HullAimed,
    RigidBodyState,
  } = getGameComponents(world);

  return function updateAim(_delta: number) {
    const eids = query(world, [ActionsQueue, Vehicle, RigidBodyState]);
    for (const ownerEid of eids) {
      if (ActionsQueue.count.get(ownerEid) === 0) continue;
      if (ActionsQueue.getKind(ownerEid, 0) !== ActionKind.Aim) continue;
      if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Finished) continue;

      const turretEid = Tank.turretEId.get(ownerEid);

      // No turret to aim → nothing we can do; finish.
      if (!turretEid) {
        ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
        continue;
      }

      // HullAimed vehicles steer the body to aim (fixed turret); the rest
      // rotate the turret. `aimerEid` is whichever entity's heading we read.
      const hullAimed = hasComponent(world, ownerEid, HullAimed);
      const aimerEid = hullAimed ? ownerEid : turretEid;
      const steer = (dir: number) =>
        hullAimed
          ? VehicleController.setRotate$(ownerEid, dir)
          : TurretController.setRotation$(turretEid, dir);

      // Resolve the target hex's world center.
      const targetCenter = MapDI.grid.hexToWorld(
        ActionsQueue.getTargetVal(ownerEid, 0, 0),
        ActionsQueue.getTargetVal(ownerEid, 0, 1),
      );

      if (!targetCenter) {
        // Off-map target — fail the action.
        steer(0);
        ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
        continue;
      }

      if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Idle) {
        ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Running);
        // Tank is stationary while aiming, so the next decision's observation
        // is already valid — open the slot immediately (§4).
        ActionsQueue.scheduleRequestNext(ownerEid, 0);
      }

      const aimerX = RigidBodyState.position.get(aimerEid, 0);
      const aimerY = RigidBodyState.position.get(aimerEid, 1);
      const desired = Math.atan2(targetCenter.y - aimerY, targetCenter.x - aimerX);
      const err = normalizeAngle(desired - RigidBodyState.rotation[aimerEid]);

      const tolerance = ActionsQueue.getParam(ownerEid, 0, AimParamOffset.tolerance) || 0.05;
      if (Math.abs(err) <= tolerance) {
        steer(0);
        ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
        continue;
      }

      // Proportional within SLOW_BAND, full speed outside it.
      const dir = Math.abs(err) >= SLOW_BAND ? Math.sign(err) : err / SLOW_BAND;
      steer(dir);
    }
  };
}
