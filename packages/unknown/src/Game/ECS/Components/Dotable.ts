import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";
import { DamageKind } from "./Damagable.ts";

/**
 * Rides a projectile: "stamp this damage-over-time on whatever part I hit".
 * The sibling of `Damagable` (instant damage) — together they fully describe
 * what a hit does. Becomes a `Dot` on the victim part.
 */
export const createDotableComponent = defineComponent((Dotable, ctx) => {
  const dps = ctx.table.flat(Float64Array);
  const kind = ctx.table.flat(Int8Array);
  const durationMs = ctx.table.flat(Float64Array);
  return {
    dps,
    kind,
    durationMs,
    addComponent(world: World, eid: EntityId, d: number, duration: number, dmgKind: DamageKind) {
      addComponent(world, eid, Dotable);
      dps.set(eid, d);
      kind.set(eid, dmgKind);
      durationMs.set(eid, duration);
    },
  };
});
