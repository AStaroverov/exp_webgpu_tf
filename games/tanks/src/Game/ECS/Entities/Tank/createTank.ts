import { createLightTank } from './Light/LightTank.ts';
import { createMediumTank } from './Medium/MediumTank.ts';
import { createHeavyTank } from './Heavy/HeavyTank.ts';
import { TankType } from '../../Components/Tank.ts';
import { EntityId } from 'bitecs';

type TankOptions =
    | { type: TankType.Light } & Parameters<typeof createLightTank>[0]
    | { type: TankType.Medium } & Parameters<typeof createMediumTank>[0]
    | { type: TankType.Heavy } & Parameters<typeof createHeavyTank>[0]

export function createTank(options: TankOptions): EntityId {
    const type = options.type;

    switch (type) {
        case TankType.Light:
            return createLightTank(options);
        case TankType.Medium:
            return createMediumTank(options);
        case TankType.Heavy:
            return createHeavyTank(options);
        default:
            throw new Error(`Unknown tank type ${ type }`);
    }
}