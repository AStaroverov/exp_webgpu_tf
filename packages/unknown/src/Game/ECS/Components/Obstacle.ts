import { World, EntityId, addComponent } from 'bitecs';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export const createObstacleComponent = defineComponent((Obstacle) => ({
    addComponent(world: World, eid: EntityId) {
        addComponent(world, eid, Obstacle);
    },
}));
