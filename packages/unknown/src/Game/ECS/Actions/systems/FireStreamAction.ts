/**
 * FireStream executor — aims the owner tank's turret at a target hex, then HOLDS
 * the shoot flag for the turret's configured window. Sibling to Fire, but the
 * stream turret has no `Firearms`: the action terminates on elapsed time, not on
 * a reload, and emission (the particle spray) is gated purely by the held flag.
 *
 * Two in-slot phases (param `phase`):
 *   1. AIMING  — rotate the turret toward the target (shared `createHexAimer`).
 *   2. HOLDING — raise the shoot flag every tick and accumulate `elapsed` (both
 *                paused while the stream charge is below the firing threshold); at
 *                `requestNextFrac · holdMs` open the slot so the next decision
 *                overlaps the tail of the hold; at `holdMs` finish, lowering the
 *                flag ONLY IF the pre-decided next action is not another
 *                FireStream (consecutive streams stay seamless — no gap).
 *
 * Acts only on slot-0 fronts of its kind; owners run concurrently.
 */

import { hasComponent, query } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import { StreamCaliberConfig } from "../../../Config/weapons.ts";
import { ActionDescriptor, encodeTarget } from "../ActionDescriptor.ts";
import { ActionHexTargetSpec, ActionKind, ActionStatus } from "../ActionTypes.ts";
import {
  FireStreamParamOffset,
  FIRE_STREAM_PHASE_AIMING,
  FIRE_STREAM_PHASE_HOLDING,
} from "../ActionSlot.ts";
import { AIM_OFF_MAP, AIM_ON_TARGET, createHexAimer } from "./FireAction.ts";

/** Enqueue spec for a FireStream action — aims at the target, then holds the stream. */
export type FireStreamActionSpec = {
  kind: ActionKind.FireStream;
  target: ActionHexTargetSpec;
};

export const FireStreamActionDescriptor: ActionDescriptor<FireStreamActionSpec> = {
  kind: ActionKind.FireStream,
  encode(eid, slot, spec) {
    const { ActionsQueue } = getGameComponents(GameDI.world);
    ActionsQueue.setKind(eid, slot, ActionKind.FireStream);
    encodeTarget(ActionsQueue, eid, slot, spec.target);
    ActionsQueue.setParam(eid, slot, FireStreamParamOffset.phase, FIRE_STREAM_PHASE_AIMING);
    ActionsQueue.setParam(eid, slot, FireStreamParamOffset.elapsed, 0);
    ActionsQueue.setParam(eid, slot, FireStreamParamOffset.targetEid, 0);
  },
  createSystem: () => createFireStreamActionSystem(),
};

export function createFireStreamActionSystem({ world } = GameDI) {
  const {
    ActionsQueue,
    Vehicle,
    Tank,
    TurretController,
    VehicleController,
    HullAimed,
    StreamFirearms,
    RigidBodyState,
  } = getGameComponents(world);
  const aimAtHexTarget = createHexAimer({ world });

  return function tick(delta: number) {
    const eids = query(world, [ActionsQueue, Vehicle, RigidBodyState]);

    for (const ownerEid of eids) {
      if (ActionsQueue.count.get(ownerEid) === 0) continue;
      if (ActionsQueue.getKind(ownerEid, 0) !== ActionKind.FireStream) continue;
      if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Finished) continue;

      const turretEid = Tank.turretEId.get(ownerEid);

      // No stream turret to hold → nothing we can do; finish (lowering a
      // flag possibly inherited raised from a previous seamless stream).
      if (!turretEid || !hasComponent(world, turretEid, StreamFirearms)) {
        if (turretEid) TurretController.setShooting$(turretEid, 0);
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

      if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Idle) {
        ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Running);
        ActionsQueue.setParam(ownerEid, 0, FireStreamParamOffset.phase, FIRE_STREAM_PHASE_AIMING);
        ActionsQueue.setParam(ownerEid, 0, FireStreamParamOffset.elapsed, 0);
      }

      const phase = ActionsQueue.getParam(ownerEid, 0, FireStreamParamOffset.phase);

      if (phase === FIRE_STREAM_PHASE_AIMING) {
        const aim = aimAtHexTarget(
          ownerEid,
          aimerEid,
          ActionsQueue.getTargetVal(ownerEid, 0, 0),
          ActionsQueue.getTargetVal(ownerEid, 0, 1),
          FireStreamParamOffset.targetEid,
          steer,
        );
        // Off-map target → can't aim; abort. The flag may be inherited
        // raised from a previous seamless stream — always lower it.
        if (aim === AIM_OFF_MAP) {
          TurretController.setShooting$(turretEid, 0);
          ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
          continue;
        }
        if (aim === AIM_ON_TARGET) {
          // On target → start the held window.
          ActionsQueue.setParam(
            ownerEid,
            0,
            FireStreamParamOffset.phase,
            FIRE_STREAM_PHASE_HOLDING,
          );
        }
        continue;
      }

      // phase === FIRE_STREAM_PHASE_HOLDING — hold the flag, count the window.
      const cfg = StreamCaliberConfig[StreamFirearms.caliberRef.get(turretEid)];

      // Charge below the firing threshold → pause the window until it recovers
      // (the flag is lowered so the held state stays honest; `elapsed` stops
      // counting). Releasing the flag is also what lets the charge regenerate.
      if (!StreamFirearms.canFire(turretEid)) {
        TurretController.setShooting$(turretEid, 0);
        ActionsQueue.scheduleRequestNext(ownerEid, 0);
        continue;
      }

      TurretController.setShooting$(turretEid, 1);

      const elapsed = ActionsQueue.getParam(ownerEid, 0, FireStreamParamOffset.elapsed) + delta;
      ActionsQueue.setParam(ownerEid, 0, FireStreamParamOffset.elapsed, elapsed);

      // Open the slot before the hold ends so the next decision is pre-decided
      // into slot 1 with no latency gap (idempotent set).
      if (elapsed >= cfg.requestNextFrac * cfg.holdMs) {
        ActionsQueue.scheduleRequestNext(ownerEid, 0);
      }

      if (elapsed >= cfg.holdMs) {
        // Consecutive FireStream → keep the flag raised: seamless stream, no gap.
        const nextIsStream =
          ActionsQueue.count.get(ownerEid) > 1 &&
          ActionsQueue.getKind(ownerEid, 1) === ActionKind.FireStream;
        if (!nextIsStream) TurretController.setShooting$(turretEid, 0);
        ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
      }
    }
  };
}
