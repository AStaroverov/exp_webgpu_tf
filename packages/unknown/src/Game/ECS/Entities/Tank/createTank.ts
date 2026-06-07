import { createLightTank } from './Light/LightTank.ts';
import { createMediumTank } from './Medium/MediumTank.ts';
import { createHeavyTank } from './Heavy/HeavyTank.ts';
import { createRanger } from './Ranger/Ranger.ts';
import { VehicleType } from '../../../Config/index.ts';
import { EntityId } from 'bitecs';

export type TankVehicleType =
    | typeof VehicleType.LightTank
    | typeof VehicleType.MediumTank
    | typeof VehicleType.HeavyTank
    | typeof VehicleType.Ranger

type TankOptions =
    | { type: typeof VehicleType.LightTank } & Parameters<typeof createLightTank>[0]
    | { type: typeof VehicleType.MediumTank } & Parameters<typeof createMediumTank>[0]
    | { type: typeof VehicleType.HeavyTank } & Parameters<typeof createHeavyTank>[0]
    | { type: typeof VehicleType.Ranger } & Parameters<typeof createRanger>[0]

export function createTank(options: TankOptions): EntityId {
    const type = options.type;

    switch (type) {
        case VehicleType.LightTank:
            return createLightTank(options);
        case VehicleType.MediumTank:
            return createMediumTank(options);
        case VehicleType.HeavyTank:
            return createHeavyTank(options);
        case VehicleType.Ranger:
            return createRanger(options);
        default:
            throw new Error(`Unknown tank type ${ type }`);
    }
}