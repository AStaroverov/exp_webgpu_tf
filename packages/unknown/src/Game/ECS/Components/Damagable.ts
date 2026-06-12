import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export enum DamageKind {
  Physical = 0,
  Fire = 1,
  Frost = 2,
  Emp = 3,
}

/**
 * Instant damage dealt on hit, tagged with its `DamageKind`. The kind travels
 * with every recorded hit through `Hitable` and triggers the kind's specialty
 * there (Frost → slow the vehicle).
 */
export const createDamagableComponent = defineComponent((Damagable, ctx) => {
  const kind = ctx.table.flat(Int8Array);
  const damage = ctx.table.flat(Float64Array);
  return {
    kind,
    damage,
    addComponent(
      world: World,
      eid: number,
      dmg: number,
      dmgKind: DamageKind = DamageKind.Physical,
    ) {
      addComponent(world, eid, Damagable);
      kind.set(eid, dmgKind);
      damage.set(eid, dmg);
    },
  };
});
