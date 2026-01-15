import { addComponent, EntityId, removeComponent, World } from 'bitecs';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

/**
 * Building component marks abandoned/ruined building entities that can be destroyed.
 * Buildings are made of destructible wall and floor pieces.
 */
export const Building = component({
    addComponent(world: World, eid: EntityId) {
        addComponent(world, eid, Building);
    },

    removeComponent(world: World, eid: EntityId) {
        removeComponent(world, eid, Building);
    },
});

/**
 * BuildingPart component marks individual building pieces (walls, floors, debris).
 */
export const BuildingPart = component({
    addComponent(world: World, eid: EntityId) {
        addComponent(world, eid, BuildingPart);
    },

    removeComponent(world: World, eid: EntityId) {
        removeComponent(world, eid, BuildingPart);
    },
});

