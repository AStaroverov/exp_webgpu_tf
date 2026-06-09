import { createLightTank } from './Light/LightTank.ts';
import { createMediumTank } from './Medium/MediumTank.ts';
import { createHeavyTank } from './Heavy/HeavyTank.ts';
import { createRocketTank } from './Rocket/RocketTank.ts';
import { VehicleType } from '../../../Config/index.ts';
import { EntityId } from 'bitecs';

export type TankVehicleType =
    | typeof VehicleType.LightTank
    | typeof VehicleType.MediumTank
    | typeof VehicleType.HeavyTank
    | typeof VehicleType.RocketTank

type TankOptions =
    | { type: typeof VehicleType.LightTank } & Parameters<typeof createLightTank>[0]
    | { type: typeof VehicleType.MediumTank } & Parameters<typeof createMediumTank>[0]
    | { type: typeof VehicleType.HeavyTank } & Parameters<typeof createHeavyTank>[0]
    | { type: typeof VehicleType.RocketTank } & Parameters<typeof createRocketTank>[0]

export function createTank(options: TankOptions): EntityId {
    const type = options.type;

    switch (type) {
        case VehicleType.LightTank:
            return createLightTank(options);
        case VehicleType.MediumTank:
            return createMediumTank(options);
        case VehicleType.HeavyTank:
            return createHeavyTank(options);
        case VehicleType.RocketTank:
            return createRocketTank(options);
        default:
            throw new Error(`Unknown tank type ${ type }`);
    }
}