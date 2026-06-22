/**
 * Stand-in decision driver — a PLACEHOLDER for the future ML policy.
 *
 * It is wired at the exact same seam the ML driver will use: it touches no
 * Action-module internals, only the documented decision API —
 * `needsDecision(eid)` (is this owner asking for its next atomic action?) and
 * `enqueueAction(eid, spec)` (hand it one) — plus the grid queries for legality.
 * Swapping this system for the real policy requires no Action-module change.
 *
 * Per tick, for every living tank that presents an open slot, it scripts ONE
 * atomic action (mimicking a single policy step):
 *   - mostly: MoveStep to a random passable neighbour cell;
 *   - occasionally: Fire at the nearest other tank's hex (Fire aims itself, then
 *     shoots one round — a single self-contained action);
 *   - if hemmed in (no passable neighbour): Hold briefly.
 *
 * Exactly one action per needsDecision hit per tick — the simplest correct
 * behaviour: needsDecision already gates on `queueDepth < MAX_QUEUE`, and
 * chaining across ticks falls out of the request-next slot mechanism.
 */

import { hasComponent, query } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { MapDI } from "../../DI/MapDI.ts";
import { getGameComponents } from "../createGameWorld.ts";
import { enqueueAction, needsDecision } from "../Actions/ActionSchedule.ts";
import { ActionKind, TargetKind } from "../Actions/ActionTypes.ts";

/** Fraction of decisions that fire (aim + shoot) instead of moving. */
const FIRE_CHANCE = 0.25;
/** Hold duration when a tank is boxed in (ms — the Hold timer accumulates delta). */
const HOLD_DURATION_MS = 1000;

export function createStandInDriverSystem({ world } = GameDI) {
  const { Tank, Vehicle, VehicleController, Children, RigidBodyState, Firearms, PlayerControlled } =
    getGameComponents(world);

  return function updateStandInDriver(_delta: number) {
    const grid = MapDI.grid;
    if (!grid) return;

    const tanks = query(world, [Tank, Vehicle, VehicleController, Children]);

    for (const eid of tanks) {
      // The human-driven tank steers its controllers directly every frame;
      // enqueuing AI actions onto it would fight those writes.
      if (hasComponent(world, eid, PlayerControlled)) continue;
      if (!needsDecision(eid)) continue;

      const px = RigidBodyState.position.get(eid, 0);
      const py = RigidBodyState.position.get(eid, 1);
      const here = grid.worldToHex(px, py);
      if (!here) continue;

      const passable = grid
        .neighbors({ q: here.q, r: here.r })
        .filter((n) => grid.isPassableFor(n.q, n.r, eid));

      if (passable.length === 0) {
        enqueueAction(eid, {
          kind: ActionKind.Hold,
          params: { duration: HOLD_DURATION_MS },
        });
        continue;
      }

      const isArmed = hasComponent(world, Tank.turretEId.get(eid), Firearms);
      if (isArmed && Math.random() < FIRE_CHANCE) {
        const target = nearestOtherTankHex(eid, px, py) ?? pickRandom(passable);
        enqueueAction(eid, {
          kind: ActionKind.Fire,
          target: { kind: TargetKind.Hex, q: target.q, r: target.r },
        });
        continue;
      }

      // Mostly: hop to a random passable neighbour.
      const dest = pickRandom(passable);
      enqueueAction(eid, {
        kind: ActionKind.MoveStep,
        target: { kind: TargetKind.Hex, q: dest.q, r: dest.r },
        params: { speed: 1 },
      });
    }
  };

  /** Hex of the closest other living tank, or null if there is none. */
  function nearestOtherTankHex(selfEid: number, px: number, py: number) {
    const tanks = query(world, [Tank, Vehicle, VehicleController, Children]);
    let bestHex: { q: number; r: number } | null = null;
    let bestDist = Infinity;
    for (const other of tanks) {
      if (other === selfEid) continue;
      const ox = RigidBodyState.position.get(other, 0);
      const oy = RigidBodyState.position.get(other, 1);
      const d = (ox - px) * (ox - px) + (oy - py) * (oy - py);
      if (d < bestDist) {
        const hex = MapDI.grid.worldToHex(ox, oy);
        if (hex) {
          bestDist = d;
          bestHex = { q: hex.q, r: hex.r };
        }
      }
    }
    return bestHex;
  }
}

function pickRandom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}
