import {
    mutatedVehicleOptions as vehicleMutatedOptions, 
    resetOptions as resetVehicleOptions,
    updateColorOptions,
    type VehicleOptions,
    type VehicleCreationOpts
} from '../Vehicle/Options.ts';

// MeleeCar-specific options that extend vehicle options
export const meleeCarOptions = {
    wheelBase: 0, // Distance between front and rear wheels
};

export const defaultMeleeCarOptions = structuredClone(meleeCarOptions);

export type MeleeCarOptions = VehicleOptions & typeof meleeCarOptions;

// Combined mutable options object for melee car creation
export const mutatedOptions: MeleeCarOptions = {
    ...vehicleMutatedOptions,
    ...meleeCarOptions,
};

export const resetOptions = (target: MeleeCarOptions, source?: Partial<VehicleCreationOpts>): MeleeCarOptions => {
    resetVehicleOptions(target, source);

    target.wheelBase = defaultMeleeCarOptions.wheelBase;

    return target;
};

export {
    updateColorOptions,
    type VehicleCreationOpts,
};

