import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const createProgressComponent = defineComponent((Progress, ctx) => {
  const age = ctx.table.flat(Float32Array);
  const maxAge = ctx.table.flat(Float32Array);
  return {
    age,
    maxAge,
    addComponent(world: World, eid: number, max: number) {
      addComponent(world, eid, Progress);
      maxAge.set(eid, max);
    },
    updateAge(eid: number, delta: number) {
      age.set(eid, age.get(eid) + delta);
    },
    getProgress(eid: number): number {
      return age.get(eid) / maxAge.get(eid);
    },
  };
});
