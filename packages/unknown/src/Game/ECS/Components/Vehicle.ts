import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';
import { VehicleType, EngineType } from '../../Config/index.ts';

export { VehicleType };

export const createVehicleComponent = defineComponent((Vehicle) => {
    const type = TypedArray.i8(delegate.defaultSize);
    const engineType = TypedArray.i8(delegate.defaultSize);
    return {
        type,
        engineType,
        addComponent(world: World, eid: EntityId, t: VehicleType) {
            addComponent(world, eid, Vehicle);
            type[eid] = t;
            engineType[eid] = 0;
        },
        setEngineType(eid: number, engine: EngineType) {
            engineType[eid] = engine;
        },
    };
});
