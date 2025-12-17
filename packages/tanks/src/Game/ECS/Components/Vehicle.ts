import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { VehicleEngineType } from '../Systems/Vehicle/VehicleControllerSystems.ts';

export enum VehicleType {
    LightTank = 0,
    MediumTank = 1,
    HeavyTank = 2,
    PlayerTank = 3,  // Special player tank - medium size but faster
    Harvester = 4,   // Bulldozer with barrier and scoop for collecting debris
}

export const Vehicle = component({
    type: TypedArray.i8(delegate.defaultSize),
    engineType: TypedArray.i8(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, type: VehicleType): void {
        addComponent(world, eid, Vehicle);
        Vehicle.type[eid] = type;
        Vehicle.engineType[eid] = 0;
    },

    setEngineType(eid: number, engineType: VehicleEngineType) {
        Vehicle.engineType[eid] = engineType;
    },
});

