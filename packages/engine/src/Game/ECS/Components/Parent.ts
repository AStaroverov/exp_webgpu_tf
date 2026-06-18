import { addComponent, hasComponent } from "bitecs";
import type { World } from "bitecs";
import { getGameComponents } from "../createGameWorld.ts";
import { defineComponent } from "renderer/src/ECS/utils.ts";

export const createParentComponent = defineComponent((Parent, ctx) => {
  const id = ctx.table.flat(Float64Array);
  return {
    id,
    addComponent(world: World, eid: number, parentEid: number) {
      addComponent(world, eid, Parent);
      id.set(eid, parentEid);

      const { Children } = getGameComponents(world);
      if (!hasComponent(world, parentEid, Children)) {
        console.warn("Parent component added to entity without Children component");
      }
    },
  };
});
