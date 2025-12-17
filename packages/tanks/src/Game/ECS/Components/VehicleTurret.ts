import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';

export const VehicleTurret = component({
    vehicleEId: TypedArray.f64(delegate.defaultSize),
    rotationSpeed: TypedArray.f32(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, vehicleEid: EntityId): void {
        addComponent(world, eid, VehicleTurret);
        VehicleTurret.vehicleEId[eid] = vehicleEid;
        VehicleTurret.rotationSpeed[eid] = 0;
    },

    setRotationSpeed(eid: EntityId, rotationSpeed: number): void {
        VehicleTurret.rotationSpeed[eid] = rotationSpeed;
    },
});

