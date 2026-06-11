import { addComponent, EntityId, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
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
export const createStreamFirearmsComponent = defineComponent((StreamFirearms) => {
  const caliberRef = TypedArray.i8(delegate.defaultSize);
  const emitAccMs = TypedArray.f64(delegate.defaultSize);
  const firedMs = TypedArray.f64(delegate.defaultSize);
  const reloading = TypedArray.f64(delegate.defaultSize);
  return {
    caliberRef,
    emitAccMs,
    firedMs,
    reloading,
    addComponent(world: World, eid: EntityId, caliber: number) {
      addComponent(world, eid, StreamFirearms);
      caliberRef[eid] = caliber;
      emitAccMs[eid] = 0;
      firedMs[eid] = 0;
      reloading[eid] = 0;
    },
    isReloading(eid: EntityId): boolean {
      return reloading[eid] > 0;
    },
    startReloading(eid: EntityId, durationMs: number) {
      reloading[eid] = durationMs;
      firedMs[eid] = 0; // the reload refills the magazine
    },
    updateReloading(eid: EntityId, dt: number) {
      reloading[eid] -= dt;
    },
  };
});
