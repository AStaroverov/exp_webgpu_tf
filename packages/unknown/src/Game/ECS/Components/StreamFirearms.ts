import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

/**
 * Marks a turret that sprays sensor particles while the shoot flag is held.
 * Presence is the "this turret streams" query — disjoint from the bullet
 * spawner (`Firearms`); a stream turret has no `Firearms` at all. Holds the
 * caliber row key + live per-entity state: the emit accumulator, the magazine
 * (`firedMs` of emission spent, capacity `fireDurationMs`) and the remaining
 * reload; all tunables are read from the global `StreamCaliberConfig` at the
 * use sites. The reload trio mirrors `Firearms`.
 */
export const createStreamFirearmsComponent = defineComponent((StreamFirearms, ctx) => {
  const caliberRef = ctx.table.flat(Int8Array);
  const emitAccMs = ctx.table.flat(Float64Array);
  const firedMs = ctx.table.flat(Float64Array);
  const reloading = ctx.table.flat(Float64Array);
  return {
    caliberRef,
    emitAccMs,
    firedMs,
    reloading,
    addComponent(world: World, eid: EntityId, caliber: number) {
      addComponent(world, eid, StreamFirearms);
      caliberRef.set(eid, caliber);
    },
    isReloading(eid: EntityId): boolean {
      return reloading.get(eid) > 0;
    },
    startReloading(eid: EntityId, durationMs: number) {
      reloading.set(eid, durationMs);
      firedMs.set(eid, 0); // the reload refills the magazine
    },
    updateReloading(eid: EntityId, dt: number) {
      reloading.set(eid, reloading.get(eid) - dt);
    },
  };
});
