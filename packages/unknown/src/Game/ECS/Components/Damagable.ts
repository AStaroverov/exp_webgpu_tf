import { addComponent, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export enum DamageKind {
  Physical = 0,
  Fire = 1,
  Frost = 2,
}

/**
 * Instant damage dealt on hit, tagged with its `DamageKind`. The kind travels
 * with every recorded hit through `Hitable` and triggers the kind's specialty
 * there (Frost → slow the vehicle).
 */
export const createDamagableComponent = defineComponent((Damagable) => {
  const damage = TypedArray.f64(delegate.defaultSize);
  const kind = TypedArray.i8(delegate.defaultSize);
  return {
    damage,
    kind,
    addComponent(
      world: World,
      eid: number,
      dmg: number,
      dmgKind: DamageKind = DamageKind.Physical,
    ) {
      addComponent(world, eid, Damagable);
      damage[eid] = dmg;
      kind[eid] = dmgKind;
    },
  };
});
