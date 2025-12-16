import { ZIndex } from '../../consts.ts';
import { CollisionGroup } from '../../Physical/createRigid.ts';
import { TankType } from './Tank.ts';

/**
 * Slot part type - determines what kind of part goes in a slot
 */
export enum SlotPartType {
    HullPart = 0,
    Caterpillar = 1,
    TurretHead = 2,
    TurretGun = 3,
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
 * Base densities for each tank type
 */
const TANK_BASE_DENSITY: Record<TankType, number> = {
    [TankType.Light]: 250,
    [TankType.Medium]: 300,
    [TankType.Heavy]: 350,
    [TankType.Player]: 300, // Same as Medium
};

/**
 * Density multipliers for each part type
 */
const PART_DENSITY_MULTIPLIER: Record<SlotPartType, number> = {
    [SlotPartType.HullPart]: 10,
    [SlotPartType.Caterpillar]: 1,
    [SlotPartType.TurretHead]: 1,
    [SlotPartType.TurretGun]: 1,
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
        shadowY: 3,
    },
    [SlotPartType.Caterpillar]: {
        z: ZIndex.TankCaterpillar,
        belongsSolverGroup: CollisionGroup.ALL,
        interactsSolverGroup: CollisionGroup.ALL,
        belongsCollisionGroup: CollisionGroup.TANK_HULL_PARTS,
        interactsCollisionGroup: CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_HULL_PARTS,
        shadowY: 3,
    },
    [SlotPartType.TurretHead]: {
        z: ZIndex.TankTurret,
        belongsSolverGroup: CollisionGroup.ALL,
        interactsSolverGroup: CollisionGroup.ALL,
        belongsCollisionGroup: CollisionGroup.TANK_TURRET_HEAD_PARTS,
        interactsCollisionGroup: CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_HEAD_PARTS | CollisionGroup.TANK_TURRET_GUN_PARTS,
        shadowY: 3,
    },
    [SlotPartType.TurretGun]: {
        z: ZIndex.TankTurret,
        belongsSolverGroup: CollisionGroup.ALL,
        interactsSolverGroup: CollisionGroup.ALL,
        belongsCollisionGroup: CollisionGroup.TANK_TURRET_GUN_PARTS,
        interactsCollisionGroup: CollisionGroup.BULLET | CollisionGroup.WALL | CollisionGroup.TANK_TURRET_HEAD_PARTS | CollisionGroup.TANK_TURRET_GUN_PARTS,
        shadowY: 4,
    },
};

/**
 * Get config for a slot part type and tank type
 */
export function getSlotPartConfig(partType: SlotPartType, tankType: TankType): SlotPartConfig {
    const baseConfig = BASE_SLOT_PART_CONFIGS[partType];
    const baseDensity = TANK_BASE_DENSITY[tankType];
    const densityMultiplier = PART_DENSITY_MULTIPLIER[partType];
    
    return {
        ...baseConfig,
        density: baseDensity * densityMultiplier,
    };
}
