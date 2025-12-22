import { addComponent, EntityId, removeComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

export const SpiceCollector = component({
    addComponent(world: World, eid: EntityId) {
        addComponent(world, eid, SpiceCollector);
    },

    removeComponent(world: World, eid: EntityId) {
        removeComponent(world, eid, SpiceCollector);
    },
});

