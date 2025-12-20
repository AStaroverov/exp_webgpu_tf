import { addComponent, EntityId, removeComponent, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

/**
 * Rock component marks rock entities that can be destroyed.
 * Rocks are made of small destructible pieces connected via joints.
 */
export const Rock = component({
    // Total number of parts in this rock
    partsCount: TypedArray.i32(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, partsCount: number) {
        addComponent(world, eid, Rock);
        Rock.partsCount[eid] = partsCount;
    },

    removeComponent(world: World, eid: EntityId) {
        removeComponent(world, eid, Rock);
        Rock.partsCount[eid] = 0;
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

