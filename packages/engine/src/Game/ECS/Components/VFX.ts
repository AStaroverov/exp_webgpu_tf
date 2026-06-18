import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

export const VFXType = {
  ExhaustSmoke: 0,
  Explosion: 1,
  HitFlash: 2,
  MuzzleFlash: 3,
  Flame: 4,
  Frost: 5,
  EmpOverlay: 6,
  EmpExplosion: 7,
} as const;

export type VFXTypeValue = (typeof VFXType)[keyof typeof VFXType];

export const createVFXComponent = defineComponent((VFX, ctx) => {
  const type = ctx.table.flat(Uint8Array);
  return {
    type,
    addComponent(world: World, eid: number, t: VFXTypeValue) {
      addComponent(world, eid, VFX);
      type.set(eid, t);
    },
  };
});
