import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';

export const VehicleTurret = component({
    rotationSpeed: TypedArray.f32(delegate.defaultSize),

    addComponent(world: World, eid: EntityId): void {
        addComponent(world, eid, VehicleTurret);
        VehicleTurret.rotationSpeed[eid] = 0;
    },

    setRotationSpeed(eid: EntityId, rotationSpeed: number): void {
        VehicleTurret.rotationSpeed[eid] = rotationSpeed;
    },
});

