import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, removeComponent, World } from 'bitecs';

/**
 * Tag component for vehicle parts (turrets, armor plates, etc.).
 */
export const VehiclePart = component({
    addComponent(world: World, eid: EntityId): void {
        addComponent(world, eid, VehiclePart);
    },

    removeComponent(world: World, eid: EntityId): void {
        removeComponent(world, eid, VehiclePart);
    },
});

// Tag component for caterpillar parts (used for track animation)
export const VehiclePartCaterpillar = component({
    addComponent(world: World, eid: EntityId): void {
        addComponent(world, eid, VehiclePartCaterpillar);
    },
    removeComponent(world: World, eid: EntityId): void {
        removeComponent(world, eid, VehiclePartCaterpillar);
    },
});

