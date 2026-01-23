import { World, EntityId, addComponent } from "bitecs";
import { component } from "renderer/src/ECS/utils";

export const Obstacle = component({
    addComponent(world: World, eid: EntityId) {
        addComponent(world, eid, Obstacle);
    },
});