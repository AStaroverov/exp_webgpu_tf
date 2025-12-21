import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { VehicleType, EngineType } from '../../Config/index.ts';

export { VehicleType };

export const Vehicle = component({
    type: TypedArray.i8(delegate.defaultSize),
    engineType: TypedArray.i8(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, type: VehicleType): void {
        addComponent(world, eid, Vehicle);
        Vehicle.type[eid] = type;
        Vehicle.engineType[eid] = 0;
    },

    setEngineType(eid: number, engineType: EngineType) {
        Vehicle.engineType[eid] = engineType;
    },
});
