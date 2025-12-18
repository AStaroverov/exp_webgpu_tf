import { createVehicleBase } from '../Vehicle/VehicleBase.ts';
import { type MeleeCarOptions } from './Options.ts';

export function createMeleeCarBase(options: MeleeCarOptions): [number, number] {
    const [vehicleEid, vehiclePid] = createVehicleBase(options);

    // MeleeCar doesn't have caterpillars or turret - just a simple ramming vehicle
    // No Tank component needed since there's no turret

    return [vehicleEid, vehiclePid];
}

