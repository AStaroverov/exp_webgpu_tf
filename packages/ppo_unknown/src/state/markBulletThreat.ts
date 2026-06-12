/**
 * markBulletThreat — fill one observer's `UnderFire` board plane.
 *
 * Two passes write the same `UnderFire` channel:
 *
 * 1. LIVE BULLETS. A bullet flies a FIXED distance (`DestroyByDistance.maxDistanceSq`)
 *    along a straight line from its `origin`, so its entire remaining path is known.
 *    For every live ENEMY bullet (threat is enemy fire, POV-relative), we walk the
 *    stretch the bullet has NOT yet crossed — from its current position to its
 *    endpoint — and mark each hex it will pass through as under fire (0/1). The
 *    bullet's current cell is not special-cased as the threat; the cells AHEAD are.
 *    The walk stops at the first cell physically blocking the shot (a `Unit` or
 *    `Obstacle`) — the bullet hits it — and at the grid edge. The cell the bullet
 *    currently sits in is never treated as a blocker (it just left the muzzle).
 *    Straight-line walk via `grid.raycast`; no honeycomb allocation.
 *
 * 2. PREDICTED FIRE. An enemy that has a queued `Fire` action (front slot 0, or
 *    pending slot 1) telegraphs where it is about to shoot. The Fire action's target
 *    is a PRECISE ring hex (rings 1..FIRE_RING_RADIUS); the shot flies the straight
 *    world line from the owner through that hex's centre (and past it). We walk that
 *    line from the enemy's hex, bounded by the vehicle's bullet flight distance and
 *    stopping at the first `Unit`/`Obstacle` (skipping `Reserved`), and mark it
 *    `UnderFire` too. The projection uses the queued Fire target hex, NOT the
 *    turret's current world heading (nor the executor's aim-lock refinement onto a
 *    unit near the hex — centre-line is a close, cheap approximation). A bare `Aim`
 *    is never projected; only a queued `Fire`.
 *
 * Marks are written into the observer's EGOCENTRIC window (axial deltas from
 * `(selfQ, selfR)`, see board.ts); path cells beyond the view radius are walked
 * (the shot still flies there) but not marked (not visible).
 */

import { query } from "bitecs";
import type { World } from "bitecs";
import { GameDI } from "../../../unknown/src/Game/DI/GameDI.ts";
import { getGameComponents } from "../../../unknown/src/Game/ECS/createGameWorld.ts";
import { ActionKind } from "../../../unknown/src/Game/ECS/Actions/ActionTypes.ts";
import { OccupantKind, type HexCell, type HexGrid } from "../../../unknown/src/Game/Map/HexGrid.ts";
import { getTankConfig, type VehicleType } from "../../../unknown/src/Game/Config/vehicles.ts";
import { BulletCaliberConfig } from "../../../unknown/src/Game/Config/weapons.ts";
import {
  BoardChannel,
  ensureUnknownInputBoard,
  hexDeltaDistance,
  VIEW_RADIUS,
  type UnknownInputBoardComponent,
} from "./board.ts";

/** Below this travelled distance the direction comes from velocity, not (pos − origin). */
const ORIGIN_EPS = 1e-3;

/**
 * Bullet flight distance (world units) for a vehicle type, or 0 if it has no gun:
 * `BulletCaliberConfig[…].maxDistance`. Gunless vehicles (non-tanks: no config) never fire.
 */
function bulletMaxDistance(type: VehicleType): number {
  const config = getTankConfig(type);
  if (config?.gun === undefined) return 0;
  return BulletCaliberConfig[config.gun.caliber].maxDistance;
}

export function markBulletThreat(
  selfEid: number,
  myTeam: number,
  selfQ: number,
  selfR: number,
  grid: HexGrid,
  world: World = GameDI.world,
): void {
  const { Bullet, RigidBodyState, DestroyByDistance, TeamRef, Vehicle, ActionsQueue } =
    getGameComponents(world);
  const UnknownInputBoard = ensureUnknownInputBoard(world);
  const bullets = query(world, [Bullet, RigidBodyState, DestroyByDistance, TeamRef]);

  for (let i = 0; i < bullets.length; i++) {
    const eid = bullets[i];
    if (TeamRef.id.get(eid) === myTeam) continue; // only enemy fire is a threat

    const px = RigidBodyState.position.get(eid, 0);
    const py = RigidBodyState.position.get(eid, 1);
    const ox = DestroyByDistance.origin.get(eid, 0);
    const oy = DestroyByDistance.origin.get(eid, 1);

    const dx = px - ox;
    const dy = py - oy;
    const travelled = Math.hypot(dx, dy);

    // Direction: along (pos − origin) once moving, else along velocity at spawn.
    let dirX: number;
    let dirY: number;
    if (travelled > ORIGIN_EPS) {
      dirX = dx / travelled;
      dirY = dy / travelled;
    } else {
      const vx = RigidBodyState.linvel.get(eid, 0);
      const vy = RigidBodyState.linvel.get(eid, 1);
      const speed = Math.hypot(vx, vy);
      if (speed <= ORIGIN_EPS) continue; // unknown heading → skip
      dirX = vx / speed;
      dirY = vy / speed;
    }

    const maxDist = Math.sqrt(DestroyByDistance.maxDistanceSq.get(eid));
    const remaining = maxDist - travelled;
    if (remaining <= 0) continue;

    // Walk the remaining path, marking each distinct hex it crosses. The current
    // cell (isFirst) is marked too — the bullet sits in it — but never blocks.
    grid.raycast(
      px,
      py,
      dirX,
      dirY,
      remaining,
      threatRayVisit(UnknownInputBoard, selfEid, selfQ, selfR, true),
    );
  }

  // ── Pass 2: predicted fire from enemies with a queued Fire action ──
  const vehicles = query(world, [Vehicle, ActionsQueue, RigidBodyState, TeamRef]);

  for (let i = 0; i < vehicles.length; i++) {
    const eid = vehicles[i];
    if (TeamRef.id.get(eid) === myTeam) continue; // only enemy fire is a threat

    const maxDist = bulletMaxDistance(Vehicle.type.get(eid) as VehicleType);
    if (maxDist <= 0) continue; // gunless (non-tank) — never fires

    // The enemy's current hex (origin of the projected ray).
    const here = grid.worldToHex(
      RigidBodyState.position.get(eid, 0),
      RigidBodyState.position.get(eid, 1),
    );
    if (!here) continue;

    const count = ActionsQueue.count.get(eid);
    // Scan the live queue slots (0 = front, 1 = pre-decided next) for a Fire.
    for (let slot = 0; slot < count; slot++) {
      if (ActionsQueue.getKind(eid, slot) !== ActionKind.Fire) continue;

      // Fire's target is a precise ring hex; the shot's line passes through its centre.
      const targetQ = ActionsQueue.getTargetVal(eid, slot, 0);
      const targetR = ActionsQueue.getTargetVal(eid, slot, 1);
      const dq = targetQ - here.q;
      const dr = targetR - here.r;
      if (dq === 0 && dr === 0) continue; // no direction

      // Walk the straight world line from the owner through the target hex centre
      // (skip `Reserved`, stop at the first `Unit`/`Obstacle` or the grid edge).
      // Bounded by the bullet's world flight distance (centre-to-centre from the
      // firing hex).
      markPredictedFireRay(
        UnknownInputBoard,
        selfEid,
        here.q,
        here.r,
        dq,
        dr,
        maxDist,
        selfQ,
        selfR,
        grid,
      );
    }
  }
}

/**
 * Visit callback for `grid.raycast` that marks each in-window cell `UnderFire = 1`
 * and stops the walk at the first blocking body. Shared by both passes: a straight
 * world-space ray over hexes is the same walk whether it carries a live bullet or a
 * projected shot.
 *
 * `markFirst` controls the starting cell: a live bullet sits in its current cell, so
 * that cell is marked (true); a projected shot starts from the firing hex, which is
 * the shooter's own cell and is left unmarked (false). The starting cell never blocks
 * either way — `Reserved` cells are transparent, `Unit`/`Obstacle` stop the shot.
 */
function threatRayVisit(
  UnknownInputBoard: UnknownInputBoardComponent,
  selfEid: number,
  selfQ: number,
  selfR: number,
  markFirst: boolean,
) {
  return (cell: HexCell, hex: { q: number; r: number }, isFirst: boolean): boolean => {
    if (markFirst || !isFirst) {
      const dq = hex.q - selfQ;
      const dr = hex.r - selfR;
      if (hexDeltaDistance(dq, dr) <= VIEW_RADIUS) {
        UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.UnderFire, 1);
      }
    }
    return (
      isFirst ||
      (cell.occupantKind !== OccupantKind.Unit && cell.occupantKind !== OccupantKind.Obstacle)
    );
  };
}

/**
 * Walk the projected fire ray from `(fromQ, fromR)` along axial delta `(dq, dr)`
 * (the offset to the queued Fire's target hex) and mark in-window cells
 * `UnderFire = 1`: a world-space `raycast` along the line through the target hex's
 * centre visits every hex the shot crosses. Bounded by `maxDist` world units
 * centre-to-centre from the firing hex (the raycast starts at that centre).
 */
function markPredictedFireRay(
  UnknownInputBoard: UnknownInputBoardComponent,
  selfEid: number,
  fromQ: number,
  fromR: number,
  dq: number,
  dr: number,
  maxDist: number,
  selfQ: number,
  selfR: number,
  grid: HexGrid,
): void {
  const origin = grid.hexToWorld(fromQ, fromR);
  const next = grid.hexToWorld(fromQ + dq, fromR + dr);
  if (!origin || !next) return;

  const dirX = next.x - origin.x;
  const dirY = next.y - origin.y;
  const len = Math.hypot(dirX, dirY);
  if (len === 0) return;

  grid.raycast(
    origin.x,
    origin.y,
    dirX / len,
    dirY / len,
    maxDist,
    threatRayVisit(UnknownInputBoard, selfEid, selfQ, selfR, false),
  );
}
