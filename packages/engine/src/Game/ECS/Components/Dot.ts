import type { EntityId } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";
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
  const dps = ctx.table.flat(Float64Array);
  const kind = ctx.table.flat(Int8Array);
  const expiry = createExpirySubComponent(Dot, ctx);
  return {
    dps,
    kind,
    ...expiry,
    refresh(eid: EntityId, dmgKind: DamageKind, damage: number, durationMs: number) {
      expiry.refresh(eid, durationMs); // adds the component → the row exists below
      dps.set(eid, damage);
      kind.set(eid, dmgKind);
    },
  };
});
