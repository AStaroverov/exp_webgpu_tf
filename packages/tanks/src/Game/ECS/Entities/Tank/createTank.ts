import { createLightTank } from './Light/LightTank.ts';
import { createMediumTank } from './Medium/MediumTank.ts';
import { createHeavyTank } from './Heavy/HeavyTank.ts';
import { createPlayerTank } from './Player/PlayerTank.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { EntityId } from 'bitecs';

export type TankVehicleType = VehicleType.LightTank | VehicleType.MediumTank | VehicleType.HeavyTank | VehicleType.PlayerTank;

type TankOptions =
    | { type: VehicleType.LightTank } & Parameters<typeof createLightTank>[0]
    | { type: VehicleType.MediumTank } & Parameters<typeof createMediumTank>[0]
    | { type: VehicleType.HeavyTank } & Parameters<typeof createHeavyTank>[0]
    | { type: VehicleType.PlayerTank } & Parameters<typeof createPlayerTank>[0]

export function createTank(options: TankOptions): EntityId {
    const type = options.type;

    switch (type) {
        case VehicleType.LightTank:
            return createLightTank(options);
        case VehicleType.MediumTank:
            return createMediumTank(options);
        case VehicleType.HeavyTank:
            return createHeavyTank(options);
        case VehicleType.PlayerTank:
            return createPlayerTank(options);
        default:
            throw new Error(`Unknown tank type ${ type }`);
    }
}