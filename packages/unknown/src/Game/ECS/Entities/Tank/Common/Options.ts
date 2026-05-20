import { BulletCaliber } from '../../../Components/Bullet.ts';
import {
    mutatedVehicleOptions as vehicleMutatedOptions,
    resetOptions as resetVehicleOptions,
    updateColorOptions,
    type VehicleOptions,
    type VehicleCreationOpts,
} from '../../Vehicle/Options.ts';

// Tank-specific options that extend vehicle options
export const tankOptions = {
    trackLength: 0,

    turret: {
        rotationSpeed: 0,
        gunWidth: 0,
        gunHeight: 0,
    },

    firearms: {
        reloadingDuration: 0,
        bulletStartPosition: [0, 0] as number[],
        bulletCaliber: BulletCaliber.Light,
    },
};

export const defaultTankOptions = structuredClone(tankOptions);

export type TankOptions = VehicleOptions & typeof tankOptions;

// Combined mutable options object for tank creation
export const mutatedOptions: TankOptions = {
    ...vehicleMutatedOptions,
    ...tankOptions,
};

export const resetOptions = (target: TankOptions, source?: Partial<VehicleCreationOpts>): TankOptions => {
    resetVehicleOptions(target, source);

    target.trackLength = defaultTankOptions.trackLength;
    target.turret.rotationSpeed = defaultTankOptions.turret.rotationSpeed;
    target.turret.gunWidth = defaultTankOptions.turret.gunWidth;
    target.turret.gunHeight = defaultTankOptions.turret.gunHeight;
    target.firearms.reloadingDuration = defaultTankOptions.firearms.reloadingDuration;
    target.firearms.bulletStartPosition = defaultTankOptions.firearms.bulletStartPosition;
    target.firearms.bulletCaliber = defaultTankOptions.firearms.bulletCaliber;

    return target;
};

export {
    updateColorOptions,
    type VehicleCreationOpts,
};
