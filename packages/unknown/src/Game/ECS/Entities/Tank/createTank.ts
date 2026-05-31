import { createLightTank } from './Light/LightTank.ts';
import { createMediumTank } from './Medium/MediumTank.ts';
import { createHeavyTank } from './Heavy/HeavyTank.ts';
import { createPlayerTank } from './Player/PlayerTank.ts';
import { VehicleType } from '../../../Config/index.ts';
import { EntityId } from 'bitecs';
import { Worlds } from '../../../DI/Worlds.ts';

export type TankVehicleType = 
    | typeof VehicleType.LightTank 
    | typeof VehicleType.MediumTank 
    | typeof VehicleType.HeavyTank 
    | typeof VehicleType.PlayerTank;

type TankOptions =
    | { type: typeof VehicleType.LightTank } & Parameters<typeof createLightTank>[2]
    | { type: typeof VehicleType.MediumTank } & Parameters<typeof createMediumTank>[2]
    | { type: typeof VehicleType.HeavyTank } & Parameters<typeof createHeavyTank>[2]
    | { type: typeof VehicleType.PlayerTank } & Parameters<typeof createPlayerTank>[2]

export function createTank(options: TankOptions, { physicsWorld: world, physicalWorld } = Worlds): EntityId {
    const type = options.type;

    switch (type) {
        case VehicleType.LightTank:
            return createLightTank(world, physicalWorld, options);
        case VehicleType.MediumTank:
            return createMediumTank(world, physicalWorld, options);
        case VehicleType.HeavyTank:
            return createHeavyTank(world, physicalWorld, options);
        case VehicleType.PlayerTank:
            return createPlayerTank(world, physicalWorld, options);
        default:
            throw new Error(`Unknown tank type ${ type }`);
    }
}