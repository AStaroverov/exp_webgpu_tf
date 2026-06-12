import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
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
export const createDamagableComponent = defineComponent((Damagable) => {
  const kind = TypedArray.i8(delegate.defaultSize);
  const damage = TypedArray.f64(delegate.defaultSize);
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
      kind[eid] = dmgKind;
      damage[eid] = dmg;
    },
  };
});
