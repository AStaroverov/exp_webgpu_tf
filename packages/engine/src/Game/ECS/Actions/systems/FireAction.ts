/**
 * Fire executor — aims the owner tank's turret at a target, then fires exactly one
 * round. A self-contained "shoot at this cell" action (target like Aim).
 *
 * Three in-slot phases (param `phase`):
 *   1. AIMING     — rotate the turret toward the target (proportional slow-down near
 *                   the goal); advance once heading error ≤ TOLERANCE. The hex target
 *                   is a *precise cell*: an enemy in it (or, failing that, the enemy
 *                   nearest to its centre in the surrounding ring) is locked and
 *                   tracked until the shot; with no enemy around, the hex centre.
 *   2. WAIT_READY — wait until the weapon is not reloading, then raise the shoot flag
 *                   (the bullet spawner, later this tick, fires the round + starts reload).
 *   3. FIRING     — the freshly started reload confirms the shot; lower the flag, finish.
 *
 * Acts only on slot-0 fronts of its kind; owners run concurrently.
 */

import { entityExists, hasComponent, query } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { MapDI } from "../../../DI/MapDI.ts";
import { normalizeAngle } from "../../../../../../../lib/math.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import { OccupantKind } from "../../../Map/HexGrid.ts";
import { ActionDescriptor, encodeTarget } from "../ActionDescriptor.ts";
import { ActionHexTargetSpec, ActionKind, ActionStatus } from "../ActionTypes.ts";
import {
  FireParamOffset,
  FIRE_PHASE_AIMING,
  FIRE_PHASE_WAIT_READY,
  FIRE_PHASE_FIRING,
} from "../ActionSlot.ts";

/** Heading-error band (rad) below which we steer proportionally instead of full speed. */
const SLOW_BAND = 0.3;
/** Heading error (rad) within which the turret counts as on-target and may fire. */
const TOLERANCE = 0.05;

/** `aimAtHexTarget` outcomes. */
export const AIM_OFF_MAP = -1;
export const AIM_TURNING = 0;
export const AIM_ON_TARGET = 1;

/**
 * Shared hex-target aimer (used by Fire and FireStream). The hex target is a
 * *precise cell* (mechanics: 3 radii of fire targets):
 *   • an enemy unit IN the hex → lock it and aim at its live body centre until
 *     the shot ("аимить именно в нее до выстрела");
 *   • no unit in the hex → the enemy in the surrounding ring (the hex's 6
 *     neighbours) nearest to the hex CENTRE is locked instead;
 *   • no enemy at all → aim at the hex centre.
 * The lock lives in the slot's `targetEidParam` (slot 0): it is resolved once
 * and re-resolved only while empty or when the locked unit is destroyed, so a
 * locked unit is tracked even if it leaves the hex.
 * Steers via the provided callback (proportional slow-down within SLOW_BAND,
 * zeroed on abort/on-target) and returns an AIM_* outcome.
 */
export function createHexAimer({ world }: Pick<typeof GameDI, "world"> = GameDI) {
  const { ActionsQueue, RigidBodyState, Vehicle, TeamRef } = getGameComponents(world);

  const sharedTarget = { q: 0, r: 0 };

  /** The locked unit still exists and is still a vehicle (eids get recycled). */
  function isAliveVehicle(eid: number): boolean {
    return eid !== 0 && entityExists(world, eid) && hasComponent(world, eid, Vehicle);
  }

  function resolveTargetEid(
    ownerEid: number,
    targetQ: number,
    targetR: number,
    centerX: number,
    centerY: number,
  ): number {
    const grid = MapDI.grid;
    const myTeam = TeamRef.id.get(ownerEid);

    // An enemy in the target hex itself wins outright.
    const inHex = grid.getOccupant(targetQ, targetR);
    if (inHex && inHex.kind === OccupantKind.Unit && TeamRef.id.get(inHex.eid) !== myTeam) {
      return inHex.eid;
    }

    // Otherwise the nearest ring: the enemy whose body centre is closest to
    // the TARGET HEX centre among the hex's 6 neighbours.
    sharedTarget.q = targetQ;
    sharedTarget.r = targetR;
    let best = 0;
    let bestDistSq = Infinity;
    for (let dir = 0; dir < 6; dir++) {
      const n = grid.neighborAt(sharedTarget, dir);
      if (!n) continue;
      const occupant = grid.getOccupant(n.q, n.r);
      if (!occupant || occupant.kind !== OccupantKind.Unit) continue;
      if (TeamRef.id.get(occupant.eid) === myTeam) continue;
      const dx = RigidBodyState.position.get(occupant.eid, 0) - centerX;
      const dy = RigidBodyState.position.get(occupant.eid, 1) - centerY;
      const distSq = dx * dx + dy * dy;
      if (distSq < bestDistSq) {
        bestDistSq = distSq;
        best = occupant.eid;
      }
    }
    return best;
  }

  return function aimAtHexTarget(
    ownerEid: number,
    aimerEid: number,
    targetQ: number,
    targetR: number,
    targetEidParam: number,
    steer: (dir: number) => void,
  ): number {
    // Resolve the target hex's world center.
    const targetCenter = MapDI.grid.hexToWorld(targetQ, targetR);

    // Off-map target → can't aim.
    if (!targetCenter) {
      steer(0);
      return AIM_OFF_MAP;
    }

    // Lock-on: keep the resolved unit until the shot; re-resolve only while
    // there is none (a unit may enter the hex mid-aim) or it was destroyed.
    let targetEid = ActionsQueue.getParam(ownerEid, 0, targetEidParam);
    if (!isAliveVehicle(targetEid)) {
      targetEid = resolveTargetEid(ownerEid, targetQ, targetR, targetCenter.x, targetCenter.y);
      ActionsQueue.setParam(ownerEid, 0, targetEidParam, targetEid);
    }

    // A locked unit is aimed at its LIVE body centre (tracks movement); with
    // no unit around, the shot goes to the hex centre.
    const aimX = targetEid !== 0 ? RigidBodyState.position.get(targetEid, 0) : targetCenter.x;
    const aimY = targetEid !== 0 ? RigidBodyState.position.get(targetEid, 1) : targetCenter.y;

    const aimerX = RigidBodyState.position.get(aimerEid, 0);
    const aimerY = RigidBodyState.position.get(aimerEid, 1);
    const desired = Math.atan2(aimY - aimerY, aimX - aimerX);
    const err = normalizeAngle(desired - RigidBodyState.rotation[aimerEid]);

    if (Math.abs(err) <= TOLERANCE) {
      steer(0);
      return AIM_ON_TARGET;
    }

    // Proportional within SLOW_BAND, full speed outside it.
    steer(Math.abs(err) >= SLOW_BAND ? Math.sign(err) : err / SLOW_BAND);
    return AIM_TURNING;
  };
}

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
    ActionsQueue.setParam(eid, slot, FireParamOffset.targetEid, 0);
  },
  createSystem: () => createFireActionSystem(),
};

export function createFireActionSystem({ world } = GameDI) {
  const {
    ActionsQueue,
    Vehicle,
    Tank,
    TurretController,
    VehicleController,
    HullAimed,
    Firearms,
    RigidBodyState,
  } = getGameComponents(world);
  const aimAtHexTarget = createHexAimer({ world });

  function tick(_delta: number) {
    const eids = query(world, [ActionsQueue, Vehicle, RigidBodyState]);

    for (const ownerEid of eids) {
      if (ActionsQueue.count.get(ownerEid) === 0) continue;
      if (ActionsQueue.getKind(ownerEid, 0) !== ActionKind.Fire) continue;
      if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Finished) continue;

      const turretEid = Tank.turretEId.get(ownerEid);

      // No turret to fire from → finish immediately.
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

      if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Idle) {
        ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Running);
        ActionsQueue.setParam(ownerEid, 0, FireParamOffset.phase, FIRE_PHASE_AIMING);
        ActionsQueue.scheduleRequestNext(ownerEid, 0);
      }

      const phase = ActionsQueue.getParam(ownerEid, 0, FireParamOffset.phase);

      if (phase === FIRE_PHASE_AIMING) {
        const aim = aimAtHexTarget(
          ownerEid,
          aimerEid,
          ActionsQueue.getTargetVal(ownerEid, 0, 0),
          ActionsQueue.getTargetVal(ownerEid, 0, 1),
          FireParamOffset.targetEid,
          steer,
        );
        // Off-map target → can't aim; abort.
        if (aim === AIM_OFF_MAP) {
          ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
          continue;
        }
        if (aim === AIM_ON_TARGET) {
          // On target → wait for the weapon, then fire a round.
          ActionsQueue.setParam(ownerEid, 0, FireParamOffset.phase, FIRE_PHASE_WAIT_READY);
        }
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
  }

  return tick;
}
