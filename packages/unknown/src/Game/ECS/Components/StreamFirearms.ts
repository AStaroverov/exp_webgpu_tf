import { addComponent, EntityId, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

/**
 * Marks a turret that sprays sensor particles while the shoot flag is held.
 * Presence is the "this turret streams" query — disjoint from the bullet
 * spawner (`Firearms`); a stream turret has no `Firearms` at all. Holds only
 * the caliber row key + the emit accumulator; all tunables are read from the
 * global `StreamCaliberConfig` at the use sites.
 */
export const createStreamFirearmsComponent = defineComponent((StreamFirearms) => {
  const caliberRef = TypedArray.i8(delegate.defaultSize);
  const emitAccMs = TypedArray.f64(delegate.defaultSize);
  return {
    caliberRef,
    emitAccMs,
    addComponent(world: World, eid: EntityId, caliber: number) {
      addComponent(world, eid, StreamFirearms);
      caliberRef[eid] = caliber;
      emitAccMs[eid] = 0;
    },
  };
});
