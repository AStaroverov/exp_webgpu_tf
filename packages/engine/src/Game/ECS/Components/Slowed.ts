import { addComponent, hasComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

/**
 * Presence = "this vehicle is slowed". `slowMul` ∈ [0, 1] is the freeze
 * amount: 0 = full speed (the dense default), 1 = fully frozen; the speed
 * sites scale movement impulse and turret turn speed by `1 - slowMul`.
 * Every Frost-kind damage event adds to it (cap 1); it thaws back each tick
 * in `createSlowedExpirySystem`, which removes the component at 0.
 */
export const createSlowedComponent = defineComponent((Slowed, ctx) => {
  const slowMul = ctx.table.flat(Float64Array);
  return {
    slowMul,
    addContribution(world: World, eid: EntityId, freeze: number) {
      if (!hasComponent(world, eid, Slowed)) {
        addComponent(world, eid, Slowed); // creates the row zeroed
      }
      slowMul.set(eid, Math.min(1, slowMul.get(eid) + freeze));
    },
  };
});
