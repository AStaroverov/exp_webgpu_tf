import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

/**
 * Existence-based save-slot for a tinted part: snapshot of the part's
 * pre-tint live color. Presence = "this part is recolored"; the tint system
 * restores from here and removes the component when no status remains.
 */
export const createOriginalColorComponent = defineComponent((OriginalColor, ctx) => {
  const r = ctx.table.flat(Float64Array);
  const g = ctx.table.flat(Float64Array);
  const b = ctx.table.flat(Float64Array);
  return {
    r,
    g,
    b,
    addComponent(world: World, eid: EntityId, cr: number, cg: number, cb: number) {
      addComponent(world, eid, OriginalColor);
      r.set(eid, cr);
      g.set(eid, cg);
      b.set(eid, cb);
    },
  };
});
