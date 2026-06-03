import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export const createVehicleTurretComponent = defineComponent((VehicleTurret) => {
    const rotationSpeed = TypedArray.f32(delegate.defaultSize);
    return {
        rotationSpeed,
        addComponent(world: World, eid: EntityId, speed: number) {
            addComponent(world, eid, VehicleTurret);
            rotationSpeed[eid] = speed;
        },
    };
});
