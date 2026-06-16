import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

/** Below this fraction of full charge the stream can't fire (firing gate + RL mask). */
export const STREAM_MIN_FIRE_CHARGE = 0.2;

/**
 * Marks a turret that sprays sensor particles while the shoot flag is held.
 * Presence is the "this turret streams" query — disjoint from the bullet
 * spawner (`Firearms`); a stream turret has no `Firearms` at all. Holds the
 * caliber row key + live per-entity state: the emit accumulator and the
 * `charge` (0..1, normalized fuel). There is no discrete reload: firing drains
 * the charge (`fireDurationMs` empties it from full), and while the trigger is
 * released the charge regenerates smoothly (`reloadMs` refills it from empty).
 * Below `STREAM_MIN_FIRE_CHARGE` the gun can't fire (see `canFire`). All
 * tunables are read from the global `StreamCaliberConfig` at the use sites.
 */
export const createStreamFirearmsComponent = defineComponent((StreamFirearms, ctx) => {
  const caliberRef = ctx.table.flat(Int8Array);
  const emitAccMs = ctx.table.flat(Float64Array);
  const charge = ctx.table.flat(Float64Array);
  return {
    caliberRef,
    emitAccMs,
    charge,
    addComponent(world: World, eid: EntityId, caliber: number) {
      addComponent(world, eid, StreamFirearms);
      caliberRef.set(eid, caliber);
      charge.set(eid, 1); // start with a full charge (config-free literal)
    },
    getCharge(eid: EntityId): number {
      return charge.get(eid);
    },
    canFire(eid: EntityId): boolean {
      return charge.get(eid) >= STREAM_MIN_FIRE_CHARGE;
    },
    /** Drain while firing: `fireDurationMs` of continuous fire empties a full charge. */
    deplete(eid: EntityId, dt: number, fireDurationMs: number) {
      charge.set(eid, Math.max(0, charge.get(eid) - dt / fireDurationMs));
    },
    /** Regenerate while idle: `reloadMs` of rest refills an empty charge. */
    recharge(eid: EntityId, dt: number, reloadMs: number) {
      charge.set(eid, Math.min(1, charge.get(eid) + dt / reloadMs));
    },
  };
});
