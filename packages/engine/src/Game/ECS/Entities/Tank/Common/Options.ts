import { BulletCaliber } from "../../../Components/Bullet.ts";
import {
  mutatedVehicleOptions as vehicleMutatedOptions,
  resetOptions as resetVehicleOptions,
  updateColorOptions,
  type VehicleOptions,
  type VehicleCreationOpts,
} from "../../Vehicle/Options.ts";

// Tank-specific options that extend vehicle options
export const tankOptions = {
  trackLength: 0,

  /** Projectile spawn offset (gun tip) in the turret's local space. */
  spawnDeltaPosition: [0, 0] as number[],

  turret: {
    rotationSpeed: 0,
    gunWidth: 0,
    gunHeight: 0,
  },

  firearms: {
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

export const resetOptions = (
  target: TankOptions,
  source?: Partial<VehicleCreationOpts>,
): TankOptions => {
  resetVehicleOptions(target, source);

  target.trackLength = defaultTankOptions.trackLength;
  target.spawnDeltaPosition = defaultTankOptions.spawnDeltaPosition;
  target.turret.rotationSpeed = defaultTankOptions.turret.rotationSpeed;
  target.turret.gunWidth = defaultTankOptions.turret.gunWidth;
  target.turret.gunHeight = defaultTankOptions.turret.gunHeight;
  target.firearms.bulletCaliber = defaultTankOptions.firearms.bulletCaliber;

  return target;
};

export { updateColorOptions, type VehicleCreationOpts };
