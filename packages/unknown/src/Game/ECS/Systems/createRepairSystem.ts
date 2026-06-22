/**
 * createRepairSystem — a `Repairer` tank salvages nearby ground scrap to heal.
 *
 * Each tick, for every `Repairer` whose cooldown has elapsed and that has an empty
 * slot to refill: count the `Salvage` debris within `REPAIR_RADIUS`; once that
 * reaches `PIECES_PER_REPAIR`, consume that many (scheduled for destruction +
 * un-tagged so they aren't re-counted) and refill ONE slot, then start the
 * cooldown. Consuming the scrap is the rate limiter — a big pile heals several
 * slots over successive cooldowns; an empty field heals nothing.
 *
 * Slot refill reuses the spawn-time `fillSlot` on the first empty hull-or-turret
 * slot (the same set `getTankHealth` counts), so health ticks straight back up.
 */

import { query, hasComponent } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";
import { scheduleRemoveEntity } from "../Utils/typicalRemoveEntity.ts";
import { fillSlot } from "../Entities/Vehicle/VehicleParts.ts";
import { isSlot, isSlotEmpty } from "../Utils/SlotUtils.ts";
import { defaultVehicleOptions, resetOptions } from "../Entities/Vehicle/Options.ts";

/** World-units radius around the tank within which scrap can be picked up (~2 hexes). */
const REPAIR_RADIUS = 90;
/** Scrap pieces consumed to restore one slot. */
const PIECES_PER_REPAIR = 3;
/** Pause (ms) between successive salvages, so a pile heals gradually, not instantly. */
const COOLDOWN_MS = 200;

const REPAIR_RADIUS_SQ = REPAIR_RADIUS * REPAIR_RADIUS;
// Private, reused across calls — fillSlot copies what it needs out of it.
const repairOptions = structuredClone(defaultVehicleOptions);

export function createRepairSystem({ world } = GameDI) {
  const { Repairer, Salvage, Tank, Vehicle, Children, RigidBodyState, PlayerRef, TeamRef } =
    getGameComponents(world);

  /**
   * First empty slot ANYWHERE in the tank's part tree — descends through the
   * non-slot containers (turret → gun, and the two tracks) so gun and caterpillar
   * slots are repaired too, not just the hull/turret-head ones.
   */
  function findFirstEmptySlotDeep(rootEid: number): number | null {
    const count = Children.entitiesCount.get(rootEid);
    for (let i = 0; i < count; i++) {
      const child = Children.entitiesIds.get(rootEid, i);
      if (isSlot(child)) {
        if (isSlotEmpty(child)) return child;
      } else if (hasComponent(world, child, Children)) {
        const found = findFirstEmptySlotDeep(child); // turret / gun / track container
        if (found != null) return found;
      }
    }
    return null;
  }

  /** Refill the first empty slot anywhere on the tank, at its current pose. */
  function repairOneSlot(vehicleEid: number): boolean {
    const slotEid = findFirstEmptySlotDeep(vehicleEid);
    if (slotEid == null) return false;

    // The part is a CompoundPart (collider anchored on the body), so its render
    // pose snaps correct next frame regardless — the hull pose here is just a seed.
    resetOptions(repairOptions, {
      playerId: PlayerRef.id.get(vehicleEid),
      teamId: TeamRef.id.get(vehicleEid),
      x: RigidBodyState.position.get(vehicleEid, 0),
      y: RigidBodyState.position.get(vehicleEid, 1),
      rotation: RigidBodyState.rotation[vehicleEid],
    });
    fillSlot(slotEid, repairOptions);
    return true;
  }

  return function updateRepair(delta: number) {
    const repairers = query(world, [Repairer, Tank, Vehicle, RigidBodyState]);
    if (repairers.length === 0) return;

    const scrap = query(world, [Salvage, RigidBodyState]);

    for (let i = 0; i < repairers.length; i++) {
      const eid = repairers[i];

      const cd = Repairer.cooldown.get(eid) - delta;
      if (cd > 0) {
        Repairer.cooldown.set(eid, cd);
        continue;
      }
      Repairer.cooldown.set(eid, 0);

      const px = RigidBodyState.position.get(eid, 0);
      const py = RigidBodyState.position.get(eid, 1);

      // Collect the nearest scrap within reach (just enough to repair once).
      const nearby: number[] = [];
      for (let s = 0; s < scrap.length; s++) {
        const part = scrap[s];
        const dx = RigidBodyState.position.get(part, 0) - px;
        const dy = RigidBodyState.position.get(part, 1) - py;
        if (dx * dx + dy * dy <= REPAIR_RADIUS_SQ) {
          nearby.push(part);
          if (nearby.length >= PIECES_PER_REPAIR) break;
        }
      }
      if (nearby.length < PIECES_PER_REPAIR) continue;

      if (!repairOneSlot(eid)) continue; // nothing to heal → leave the scrap be

      for (const part of nearby) {
        scheduleRemoveEntity(part);
      }
      Repairer.cooldown.set(eid, COOLDOWN_MS);
    }
  };
}
