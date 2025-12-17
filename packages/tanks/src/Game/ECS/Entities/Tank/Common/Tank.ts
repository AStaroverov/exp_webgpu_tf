import { GameDI } from '../../../../DI/GameDI.ts';
import { Firearms } from '../../../Components/Firearms.ts';
import { Tank } from '../../../Components/Tank.ts';
import { createVehicleBase, createVehicleTurret } from '../../Vehicle/VehicleBase.ts';
import { type TankOptions } from './Options.ts';

export function createTankBase(options: TankOptions, { world } = GameDI): [number, number] {
    const [vehicleEid, vehiclePid] = createVehicleBase(options);

    // Add Tank-specific components
    Tank.addComponent(world, vehicleEid);
    Tank.setCaterpillarsLength(vehicleEid, options.caterpillarLength);

    return [vehicleEid, vehiclePid];
}

export function createTankTurret(
    options: TankOptions,
    tankEid: number,
    tankPid: number,
    { world } = GameDI,
): [number, number] {
    const [turretEid, turretPid] = createVehicleTurret(
        options,
        options.turret,
        tankEid,
        tankPid,
    );

    // Add Tank-specific components
    Tank.setTurretEid(tankEid, turretEid);

    // Add Firearms component for shooting capability
    Firearms.addComponent(world, turretEid);
    Firearms.setData(turretEid, options.firearms.bulletStartPosition, options.firearms.bulletCaliber);
    Firearms.setReloadingDuration(turretEid, options.firearms.reloadingDuration);

    return [turretEid, turretPid];
}
