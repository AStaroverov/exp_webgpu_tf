import { addComponent, World } from "bitecs";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const VFXType = {
  ExhaustSmoke: 0,
  Explosion: 1,
  HitFlash: 2,
  MuzzleFlash: 3,
  Flame: 4,
  Frost: 5,
} as const;

export type VFXTypeValue = (typeof VFXType)[keyof typeof VFXType];

export const createVFXComponent = defineComponent((VFX) => {
  const type = TypedArray.u8(delegate.defaultSize);
  return {
    type,
    addComponent(world: World, eid: number, t: VFXTypeValue) {
      addComponent(world, eid, VFX);
      type[eid] = t;
    },
  };
});
