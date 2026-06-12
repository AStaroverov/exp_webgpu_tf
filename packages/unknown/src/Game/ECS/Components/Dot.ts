import { EntityId } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";
import { createExpirySubComponent } from "./lib/createExpirySubComponent.ts";
import { DamageKind } from "./Damagable.ts";

/**
 * Active damage-over-time on a vehicle part. Presence = "this part takes dps".
 * Stamped from a projectile's `Dotable`; `createDotSystem` records the per-tick
 * damage through `Hitable.hit$` (so the kind's specialty triggers there too).
 * The countdown is the expiry sub-component: refreshed on re-stamp (max, not
 * stack), ticked down and removed by `createExpirySystem`.
 */
export const createDotComponent = defineComponent((Dot, ctx) => {
  const dps = TypedArray.f64(delegate.defaultSize);
  const kind = TypedArray.i8(delegate.defaultSize);
  const expiry = createExpirySubComponent(Dot, ctx);
  return {
    dps,
    kind,
    ...expiry,
    refresh(eid: EntityId, dmgKind: DamageKind, damage: number, durationMs: number) {
      expiry.refresh(eid, durationMs);
      dps[eid] = damage;
      kind[eid] = dmgKind;
    },
  };
});
