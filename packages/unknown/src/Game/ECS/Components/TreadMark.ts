import { addComponent, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const createTreadMarkComponent = defineComponent((TreadMark) => ({
  addComponent(world: World, eid: number) {
    addComponent(world, eid, TreadMark);
  },
}));
