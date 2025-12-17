import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, removeComponent, World } from 'bitecs';

export const VehiclePart = component({
    jointPid: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, jointPid: number): void {
        addComponent(world, eid, VehiclePart);
        VehiclePart.jointPid[eid] = jointPid;
    },

    resetComponent(eid: EntityId): void {
        VehiclePart.jointPid[eid] = 0;
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

