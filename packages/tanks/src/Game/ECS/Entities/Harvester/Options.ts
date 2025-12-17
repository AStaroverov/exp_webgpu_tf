import {
    mutatedVehicleOptions as vehicleMutatedOptions, resetOptions as resetVehicleOptions,
    updateColorOptions,
    type VehicleOptions,
    type VehicleCreationOpts
} from '../Vehicle/Options.ts';

// Harvester-specific options that extend vehicle options
export const harvesterOptions = {
    caterpillarLength: 0,

    turret: {
        rotationSpeed: 0,
    },
};

export const defaultHarvesterOptions = structuredClone(harvesterOptions);

export type HarvesterOptions = VehicleOptions & typeof harvesterOptions;

// Combined mutable options object for harvester creation
export const mutatedOptions: HarvesterOptions = {
    ...vehicleMutatedOptions,
    ...harvesterOptions,
};

export const resetOptions = (target: HarvesterOptions, source?: Partial<VehicleCreationOpts>): HarvesterOptions => {
    resetVehicleOptions(target, source);

    target.caterpillarLength = defaultHarvesterOptions.caterpillarLength;
    target.turret.rotationSpeed = defaultHarvesterOptions.turret.rotationSpeed;

    return target;
};

export {
    updateColorOptions,
    type VehicleCreationOpts,
};

