import { ZIndex } from '../../consts.ts';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { VehicleType } from './Vehicle.ts';

/**
 * Slot part type - determines what kind of part goes in a slot
 */
export enum SlotPartType {
    HullPart = 0,
    Caterpillar = 1,
    TurretHead = 2,
    TurretGun = 3,
    Barrier = 4,      // Harvester's impenetrable barrier (controlled like turret)
    Scoop = 5,        // Harvester's front scoop for collecting debris
    Shield = 6,       // Energy shield - arc of small parts, bullet-only collision, regenerates
    Wheel = 7,        // MeleeCar's wheels
    Detail = 8,       // Small decorative details
}

/**
 * Configuration for a slot part - immutable preset data
 */
export type SlotPartConfig = {
    z: number;
    density: number;
    belongsSolverGroup: number;
    interactsSolverGroup: number;
    belongsCollisionGroup: number;
    interactsCollisionGroup: number;
    shadowY: number;
}

/**
 * Base densities for each vehicle type
 */
const VEHICLE_BASE_DENSITY: Record<VehicleType, number> = {
    [VehicleType.LightTank]: 250,
    [VehicleType.MediumTank]: 300,
    [VehicleType.HeavyTank]: 350,
    [VehicleType.PlayerTank]: 300, // Same as Medium
    [VehicleType.Harvester]: 350, // Heavy like a bulldozer
    [VehicleType.MeleeCar]: 200,  // Light and fast
};

/**
 * Density multipliers for each part type
 */
const PART_DENSITY_MULTIPLIER: Record<SlotPartType, number> = {
    [SlotPartType.HullPart]: 10,
    [SlotPartType.Caterpillar]: 1,
    [SlotPartType.TurretHead]: 1,
    [SlotPartType.TurretGun]: 1,
    [SlotPartType.Barrier]: 15, // Very heavy, impenetrable barrier
    [SlotPartType.Scoop]: 8,    // Heavy scoop for pushing debris
    [SlotPartType.Shield]: 0.1, // Very light energy shield parts
    [SlotPartType.Wheel]: 2,    // Wheels - moderate density
    [SlotPartType.Detail]: 0.5, // Light decorative details
};

/**
 * Base configurations for slot part types (without density - that depends on tank type)
 */
const BASE_SLOT_PART_CONFIGS: Record<SlotPartType, Omit<SlotPartConfig, 'density'>> = {
    [SlotPartType.HullPart]: {
        z: ZIndex.TankHull,
        belongsSolverGroup: CollisionGroup.ALL,
        interactsSolverGroup: CollisionGroup.ALL,
        belongsCollisionGroup: CollisionGroup.TANK_HULL_PARTS,
        interactsCollisionGroup: CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_HULL_PARTS,
        shadowY: 3 * 3,
    },
    [SlotPartType.Caterpillar]: {
        z: ZIndex.TankCaterpillar,
        belongsSolverGroup: CollisionGroup.ALL,
        interactsSolverGroup: CollisionGroup.ALL,
        belongsCollisionGroup: CollisionGroup.TANK_HULL_PARTS,
        interactsCollisionGroup: CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_HULL_PARTS,
        shadowY: 3 * 3,
    },
    [SlotPartType.TurretHead]: {
        z: ZIndex.TankTurret,
        belongsSolverGroup: CollisionGroup.ALL,
        interactsSolverGroup: CollisionGroup.ALL,
        belongsCollisionGroup: CollisionGroup.TANK_TURRET_HEAD_PARTS,
        interactsCollisionGroup: CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_HEAD_PARTS | CollisionGroup.TANK_TURRET_GUN_PARTS,
        shadowY: 3 * 3,
    },
    [SlotPartType.TurretGun]: {
        z: ZIndex.TankTurret,
        belongsSolverGroup: CollisionGroup.ALL,
        interactsSolverGroup: CollisionGroup.ALL,
        belongsCollisionGroup: CollisionGroup.TANK_TURRET_GUN_PARTS,
        interactsCollisionGroup: CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_HEAD_PARTS | CollisionGroup.TANK_TURRET_GUN_PARTS,
        shadowY: 4 * 3,
    },
    [SlotPartType.Barrier]: {
        z: ZIndex.TankTurret,
        belongsSolverGroup: CollisionGroup.ALL,
        interactsSolverGroup: CollisionGroup.ALL,
        // Barrier blocks bullets but is impenetrable (no BULLET in interacts means no damage)
        belongsCollisionGroup: CollisionGroup.TANK_TURRET_HEAD_PARTS,
        interactsCollisionGroup: CollisionGroup.WALL | CollisionGroup.TANK_TURRET_HEAD_PARTS,
        shadowY: 4 * 3,
    },
    [SlotPartType.Scoop]: {
        z: ZIndex.TankHull,
        belongsSolverGroup: CollisionGroup.ALL,
        interactsSolverGroup: CollisionGroup.ALL,
        belongsCollisionGroup: CollisionGroup.TANK_HULL_PARTS,
        // Scoop interacts with debris to collect them
        interactsCollisionGroup: CollisionGroup.WALL | CollisionGroup.TANK_HULL_PARTS,
        shadowY: 2 * 3,
    },
    [SlotPartType.Shield]: {
        z: ZIndex.Shield,
        // Shield has no physical push effect - bullets pass through after collision
        belongsSolverGroup: CollisionGroup.ALL,
        interactsSolverGroup: CollisionGroup.ALL,
        // Shield only collides with bullets - not with each other, not with walls, not with tank parts
        belongsCollisionGroup: CollisionGroup.SHIELD,
        interactsCollisionGroup: CollisionGroup.BULLET,
        shadowY: 0 * 3, // No shadow for energy shield
    },
    [SlotPartType.Wheel]: {
        z: ZIndex.TankCaterpillar,
        belongsSolverGroup: CollisionGroup.ALL,
        interactsSolverGroup: CollisionGroup.ALL,
        belongsCollisionGroup: CollisionGroup.TANK_HULL_PARTS,
        interactsCollisionGroup: CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_HULL_PARTS,
        shadowY: 2 * 3,
    },
    [SlotPartType.Detail]: {
        z: ZIndex.TankHull,
        belongsSolverGroup: CollisionGroup.ALL,
        interactsSolverGroup: CollisionGroup.ALL,
        belongsCollisionGroup: CollisionGroup.TANK_HULL_PARTS,
        interactsCollisionGroup: CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_HULL_PARTS,
        shadowY: 2 * 3,
    },
};

/**
 * Get config for a slot part type and vehicle type
 */
export function getSlotPartConfig(partType: SlotPartType, vehicleType: VehicleType): SlotPartConfig {
    const baseConfig = BASE_SLOT_PART_CONFIGS[partType];
    const baseDensity = VEHICLE_BASE_DENSITY[vehicleType];
    const densityMultiplier = PART_DENSITY_MULTIPLIER[partType];
    
    return {
        ...baseConfig,
        density: baseDensity * densityMultiplier,
    };
}
