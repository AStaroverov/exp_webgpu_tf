import { addComponent } from "bitecs";
import type { World, EntityId } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const createObstacleComponent = defineComponent((Obstacle) => ({
  addComponent(world: World, eid: EntityId) {
    addComponent(world, eid, Obstacle);
  },
}));
