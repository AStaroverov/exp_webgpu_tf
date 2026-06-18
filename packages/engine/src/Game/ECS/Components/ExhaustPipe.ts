import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

export const createExhaustPipeComponent = defineComponent((ExhaustPipe, ctx) => {
  const relativeX = ctx.table.flat(Float32Array);
  const relativeY = ctx.table.flat(Float32Array);
  const direction = ctx.table.flat(Float32Array);
  const emissionRate = ctx.table.flat(Float32Array);
  const emissionAccumulator = ctx.table.flat(Float32Array);

  return {
    relativeX,
    relativeY,
    direction,
    emissionRate,
    emissionAccumulator,

    addComponent(world: World, eid: number, rx: number, ry: number, dir: number, rate: number) {
      addComponent(world, eid, ExhaustPipe);
      relativeX.set(eid, rx);
      relativeY.set(eid, ry);
      direction.set(eid, dir);
      emissionRate.set(eid, rate);
    },
  };
});
