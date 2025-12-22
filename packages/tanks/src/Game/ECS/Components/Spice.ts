import { addComponent, EntityId, removeComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

/**
 * Spice component marks spice entities that can be collected by harvester scoop.
 * Spices are small resources that spawn in clusters.
 */
export const Spice = component({
    addComponent(world: World, eid: EntityId) {
        addComponent(world, eid, Spice);
    },

    removeComponent(world: World, eid: EntityId) {
        removeComponent(world, eid, Spice);
    },
});
