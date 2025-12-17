import { GameDI } from '../../../DI/GameDI.ts';
import { Tank } from '../../Components/Tank.ts';
import { createVehicleBase, createVehicleTurret } from '../Vehicle/VehicleBase.ts';
import { type HarvesterOptions } from './Options.ts';

export function createHarvesterBase(options: HarvesterOptions, { world } = GameDI): [number, number] {
    const [vehicleEid, vehiclePid] = createVehicleBase(options);

    // Add Tank component for caterpillars and turret tracking
    Tank.addComponent(world, vehicleEid);
    Tank.setCaterpillarsLength(vehicleEid, options.caterpillarLength);

    return [vehicleEid, vehiclePid];
}

export function createHarvesterTurret(
    options: HarvesterOptions,
    harvesterEid: number,
    harvesterPid: number,
): [number, number] {
    const [turretEid, turretPid] = createVehicleTurret(
        options,
        options.turret,
        harvesterEid,
        harvesterPid,
    );

    // Add Tank turret reference (no Firearms - harvester doesn't shoot)
    Tank.setTurretEid(harvesterEid, turretEid);

    return [turretEid, turretPid];
}

