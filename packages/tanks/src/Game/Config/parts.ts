/**
 * Slot Parts Configuration
 * 
 * Configuration for destructible parts attached to vehicles:
 * hull parts, caterpillars, turret components, shields, etc.
 */

import { ZIndexConfig } from './zindex.ts';
import { CollisionGroupConfig } from './physics.ts';
import { VehicleType } from './vehicles.ts';

// =============================================================================
// SLOT PART TYPES
// =============================================================================

/**
 * Types of parts that can be placed in vehicle slots.
 */
export enum SlotPartType {
    /** Main hull armor pieces */
    HullPart = 0,
    /** Track/caterpillar segments */
    Caterpillar = 1,
    /** Turret head armor */
    TurretHead = 2,
    /** Turret gun barrel */
    TurretGun = 3,
    /** Harvester's impenetrable barrier */
    Barrier = 4,
    /** Harvester's front scoop */
    Scoop = 5,
    /** Energy shield (regenerates, blocks bullets only) */
    Shield = 6,
    /** Car wheels */
    Wheel = 7,
}

// =============================================================================
// DENSITY CONFIGURATION
// =============================================================================

/**
 * Base density values for each vehicle type.
 * Used as multiplier base for part densities.
 */
export const VehicleBaseDensity: Record<VehicleType, number> = {
    [VehicleType.LightTank]: 25 * 3,
    [VehicleType.MediumTank]: 30 * 3,
    [VehicleType.HeavyTank]: 35 * 3,
    [VehicleType.PlayerTank]: 30 * 3,
    [VehicleType.Harvester]: 35 * 3,
    [VehicleType.MeleeCar]: 20 * 3,
};

/**
 * Density multipliers for each part type.
 * Final density = VehicleBaseDensity[vehicleType] * PartDensityMultiplier[partType]
 */
export const PartDensityMultiplier: Record<SlotPartType, number> = {
    [SlotPartType.HullPart]: 20,
    [SlotPartType.Caterpillar]: 5,
    [SlotPartType.TurretHead]: 15,
    [SlotPartType.TurretGun]: 10,
    [SlotPartType.Barrier]: 30,    // Very heavy, impenetrable
    [SlotPartType.Scoop]: 15,       // Heavy scoop for pushing
    [SlotPartType.Shield]: 1,    // Very light energy
    [SlotPartType.Wheel]: 4,       // Moderate weight
};

// =============================================================================
// PART PHYSICS CONFIGURATION
// =============================================================================

export type SlotPartPhysics = {
    /** Z-index for rendering order */
    z: number;
    /** Solver group membership */
    belongsSolverGroup: number;
    /** Solver group interactions */
    interactsSolverGroup: number;
    /** Collision group membership */
    belongsCollisionGroup: number;
    /** Collision group interactions */
    interactsCollisionGroup: number;
};

/**
 * Physics and collision configuration for each part type.
 */
export const PartPhysicsConfig: Record<SlotPartType, SlotPartPhysics> = {
    [SlotPartType.HullPart]: {
        z: ZIndexConfig.TankHull,
        belongsSolverGroup: CollisionGroupConfig.ALL,
        interactsSolverGroup: CollisionGroupConfig.ALL,
        belongsCollisionGroup: CollisionGroupConfig.VEHICLE_HULL_PARTS,
        interactsCollisionGroup: 
            CollisionGroupConfig.BULLET | 
            CollisionGroupConfig.OBSTACLE | 
            CollisionGroupConfig.VEHICLE_HULL_PARTS |
            CollisionGroupConfig.SPICE_COLLECTOR,
    },
    
    [SlotPartType.Caterpillar]: {
        z: ZIndexConfig.TankCaterpillar,
        belongsSolverGroup: CollisionGroupConfig.ALL,
        interactsSolverGroup: CollisionGroupConfig.ALL,
        belongsCollisionGroup: CollisionGroupConfig.VEHICLE_HULL_PARTS,
        interactsCollisionGroup: 
            CollisionGroupConfig.BULLET | 
            CollisionGroupConfig.OBSTACLE | 
            CollisionGroupConfig.VEHICLE_HULL_PARTS |
            CollisionGroupConfig.SPICE_COLLECTOR,
    },
    
    [SlotPartType.TurretHead]: {
        z: ZIndexConfig.TankTurret,
        belongsSolverGroup: CollisionGroupConfig.ALL,
        interactsSolverGroup: CollisionGroupConfig.ALL,
        belongsCollisionGroup: CollisionGroupConfig.TANK_TURRET_HEAD_PARTS,
        interactsCollisionGroup: 
            CollisionGroupConfig.BULLET | 
            CollisionGroupConfig.OBSTACLE | 
            CollisionGroupConfig.TANK_TURRET_HEAD_PARTS | 
            CollisionGroupConfig.TANK_TURRET_GUN_PARTS |
            CollisionGroupConfig.SPICE_COLLECTOR,
    },
    
    [SlotPartType.TurretGun]: {
        z: ZIndexConfig.TankTurret,
        belongsSolverGroup: CollisionGroupConfig.ALL,
        interactsSolverGroup: CollisionGroupConfig.ALL,
        belongsCollisionGroup: CollisionGroupConfig.TANK_TURRET_GUN_PARTS,
        interactsCollisionGroup: 
            CollisionGroupConfig.BULLET | 
            CollisionGroupConfig.OBSTACLE | 
            CollisionGroupConfig.TANK_TURRET_HEAD_PARTS | 
            CollisionGroupConfig.TANK_TURRET_GUN_PARTS |
            CollisionGroupConfig.SPICE_COLLECTOR,
    },
    
    [SlotPartType.Barrier]: {
        z: ZIndexConfig.TankTurret,
        belongsSolverGroup: CollisionGroupConfig.ALL,
        interactsSolverGroup: CollisionGroupConfig.ALL,
        belongsCollisionGroup: CollisionGroupConfig.TANK_TURRET_HEAD_PARTS,
        interactsCollisionGroup: 
            CollisionGroupConfig.OBSTACLE | 
            CollisionGroupConfig.TANK_TURRET_HEAD_PARTS,
    },
    
    [SlotPartType.Scoop]: {
        z: ZIndexConfig.TankHull,
        belongsSolverGroup: CollisionGroupConfig.ALL,
        interactsSolverGroup: CollisionGroupConfig.ALL,
        belongsCollisionGroup: CollisionGroupConfig.VEHICLE_HULL_PARTS,
        interactsCollisionGroup: 
            CollisionGroupConfig.OBSTACLE | 
            CollisionGroupConfig.VEHICLE_HULL_PARTS |
            CollisionGroupConfig.SPICE,
    },
    
    [SlotPartType.Shield]: {
        z: ZIndexConfig.Shield,
        belongsSolverGroup: CollisionGroupConfig.ALL,
        interactsSolverGroup: CollisionGroupConfig.ALL,
        belongsCollisionGroup: CollisionGroupConfig.SHIELD,
        interactsCollisionGroup: CollisionGroupConfig.BULLET,
    },
    
    [SlotPartType.Wheel]: {
        z: ZIndexConfig.TankCaterpillar,
        belongsSolverGroup: CollisionGroupConfig.ALL,
        interactsSolverGroup: CollisionGroupConfig.ALL,
        belongsCollisionGroup: CollisionGroupConfig.VEHICLE_HULL_PARTS,
        interactsCollisionGroup: 
            CollisionGroupConfig.BULLET | 
            CollisionGroupConfig.OBSTACLE | 
            CollisionGroupConfig.VEHICLE_HULL_PARTS |
            CollisionGroupConfig.SPICE_COLLECTOR,
    },
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Calculate density for a part based on vehicle type and part type.
 */
export function getPartDensity(vehicleType: VehicleType, partType: SlotPartType): number {
    return VehicleBaseDensity[vehicleType] * PartDensityMultiplier[partType];
}

/**
 * Get complete configuration for a slot part.
 */
export function getSlotPartConfig(
    partType: SlotPartType,
    vehicleType: VehicleType
): SlotPartPhysics & { density: number } {
    return {
        ...PartPhysicsConfig[partType],
        density: getPartDensity(vehicleType, partType),
    };
}
