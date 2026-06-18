import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

export const createTurretControllerComponent = defineComponent((TurretController, ctx) => {
  const shoot = ctx.table.flat(Float32Array);
  const rotation = ctx.table.flat(Float64Array);
  return {
    shoot,
    rotation,
    addComponent(world: World, eid: number) {
      addComponent(world, eid, TurretController);
    },
    shouldShoot(eid: number): boolean {
      return shoot.get(eid) > 0;
    },
    setShooting$: ctx.obs((eid: number, v: number) => {
      shoot.set(eid, v);
    }),
    setRotation$: ctx.obs((eid: number, v: number) => {
      rotation.set(eid, v);
    }),
  };
});
