import { addComponent, EntityId, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";
import { DamageKind } from "./Damagable.ts";

/**
 * Active damage-over-time on a vehicle part. Presence = "this part takes dps".
 * Stamped from a projectile's `Dotable`; `createDotSystem` records the per-tick
 * damage through `Hitable.hit$` (so the kind's specialty triggers there too)
 * and removes the component when `remaining` runs out.
 */
export const createDotComponent = defineComponent((Dot) => {
  const dps = TypedArray.f64(delegate.defaultSize);
  const kind = TypedArray.i8(delegate.defaultSize);
  const remaining = TypedArray.f64(delegate.defaultSize);
  return {
    dps,
    kind,
    remaining,
    addComponent(world: World, eid: EntityId, d: number, durationMs: number, dmgKind: DamageKind) {
      addComponent(world, eid, Dot);
      dps[eid] = d;
      kind[eid] = dmgKind;
      remaining[eid] = durationMs;
    },
  };
});
