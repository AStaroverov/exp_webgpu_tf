import { addComponent, EntityId, removeComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

/**
 * Rock component marks rock entities that can be destroyed.
 * Rocks are made of small destructible pieces connected via joints.
 */
export const Rock = component({
    addComponent(world: World, eid: EntityId) {
        addComponent(world, eid, Rock);
    },

    removeComponent(world: World, eid: EntityId) {
        removeComponent(world, eid, Rock);
    },
});

/**
 * RockPart component marks individual rock pieces that are connected to the main rock body.
 */
export const RockPart = component({
    addComponent(world: World, eid: EntityId) {
        addComponent(world, eid, RockPart);
    },

    removeComponent(world: World, eid: EntityId) {
        removeComponent(world, eid, RockPart);
    },
});

