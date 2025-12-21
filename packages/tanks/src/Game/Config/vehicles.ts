/**
 * Vehicles Configuration
 * 
 * All vehicle types and their characteristics:
 * tanks (light, medium, heavy, player), harvester, melee car.
 */

import { PI } from '../../../../../lib/math.ts';
import { BulletCaliber, ReloadConfig, TurretSpeedConfig } from './weapons.ts';

// =============================================================================
// VEHICLE TYPES
// =============================================================================

/**
 * All available vehicle types in the game.
 */
export enum VehicleType {
    LightTank = 0,
    MediumTank = 1,
    HeavyTank = 2,
    PlayerTank = 3,  // Special player tank - medium size but faster
    Harvester = 4,   // Bulldozer with barrier and scoop for collecting debris
    MeleeCar = 5,    // Fast 4-wheeled car for ramming
}

// =============================================================================
// ENGINE TYPES
// =============================================================================

/**
 * Engine types determine movement speed and power.
 */
export enum EngineType {
    v6 = 0,
    v8 = 1,
    v12 = 2,
    v8_turbo = 3,  // Player special engine - faster than v8
}

export const EngineLabels: Record<EngineType, string> = {
    [EngineType.v6]: 'v6',
    [EngineType.v8]: 'v8',
    [EngineType.v12]: 'v12',
    [EngineType.v8_turbo]: 'v8 Turbo',
};

/**
 * Engine power multipliers relative to base values.
 */
export const EngineConfig: Record<EngineType, { impulseMult: number; rotationMult: number }> = {
    [EngineType.v6]: {
        impulseMult: 0.8,
        rotationMult: 0.9,
    },
    [EngineType.v8]: {
        impulseMult: 1.0,
        rotationMult: 1.0,
    },
    [EngineType.v12]: {
        impulseMult: 2.0,
        rotationMult: 3.0,
    },
    [EngineType.v8_turbo]: {
        impulseMult: 2.0,
        rotationMult: 2.0,
    },
};

// =============================================================================
// TANK CONFIGURATIONS
// =============================================================================

export type TankStats = {
    /** Vehicle type identifier */
    type: VehicleType;
    /** Engine type */
    engine: EngineType;
    /** Part size in pixels */
    size: number;
    /** Padding between parts */
    padding: number;
    /** Base density multiplier */
    density: number;
    /** Approximate collider radius for broad phase */
    colliderRadius: number;
    /** Hull dimensions [width, height] in padding units */
    hullSize: [number, number];
    /** Turret dimensions [width, height] in padding units */
    turretSize: [number, number];
    /** Hull parts grid [cols, rows] */
    hullGrid: [number, number];
    /** Turret head grid [cols, rows] */
    turretHeadGrid: [number, number];
    /** Turret gun grid [cols, rows] */
    turretGunGrid: [number, number];
    /** Number of caterpillar lines */
    caterpillarLines: number;
    /** Caterpillar part size */
    caterpillarSize: number;
    /** Track anchor X offset from center */
    trackAnchorXMult: number;
    /** Turret rotation speed (radians/sec) */
    turretSpeed: number;
    /** Reload duration in ms */
    reloadTime: number;
    /** Bullet caliber */
    bulletCaliber: BulletCaliber;
    /** Bullet start position Y offset multiplier */
    bulletOffsetYMult: number;
    /** Colors */
    colors: {
        tracks: Float32Array;
        turret: Float32Array;
    };
};

/**
 * Light Tank Configuration
 * Fast and agile, but weak armor and firepower.
 */
export const LightTankConfig: TankStats = {
    type: VehicleType.LightTank,
    engine: EngineType.v6,
    size: 5,
    padding: 6, // size + 1
    density: 250,
    colliderRadius: 50,
    hullSize: [8, 10],
    turretSize: [6, 6],
    hullGrid: [8, 10],
    turretHeadGrid: [5, 6],
    turretGunGrid: [2, 6],
    caterpillarLines: 12,
    caterpillarSize: 6, // size - 1
    trackAnchorXMult: 4,
    turretSpeed: TurretSpeedConfig.light,
    reloadTime: ReloadConfig.light,
    bulletCaliber: BulletCaliber.Light,
    bulletOffsetYMult: 9,
    colors: {
        tracks: new Float32Array([0.6, 0.6, 0.6, 1]),
        turret: new Float32Array([0.6, 1, 0.6, 1]),
    },
};

/**
 * Medium Tank Configuration
 * Balanced stats across all areas.
 */
export const MediumTankConfig: TankStats = {
    type: VehicleType.MediumTank,
    engine: EngineType.v8,
    size: 6,
    padding: 7, // size + 1
    density: 300,
    colliderRadius: 80,
    hullSize: [8, 12],
    turretSize: [8, 8],
    hullGrid: [8, 11],
    turretHeadGrid: [6, 7],
    turretGunGrid: [2, 10],
    caterpillarLines: 22,
    caterpillarSize: 3,
    trackAnchorXMult: 5,
    turretSpeed: TurretSpeedConfig.medium,
    reloadTime: ReloadConfig.medium,
    bulletCaliber: BulletCaliber.Medium,
    bulletOffsetYMult: 13,
    colors: {
        tracks: new Float32Array([0.5, 0.5, 0.5, 1]),
        turret: new Float32Array([0.5, 1, 0.5, 1]),
    },
};

/**
 * Heavy Tank Configuration
 * Slow but powerful, heavy armor.
 */
export const HeavyTankConfig: TankStats = {
    type: VehicleType.HeavyTank,
    engine: EngineType.v12,
    size: 8,
    padding: 9, // size + 1
    density: 350,
    colliderRadius: 150,
    hullSize: [10, 14],
    turretSize: [8, 8],
    hullGrid: [10, 14],
    turretHeadGrid: [7, 9],
    turretGunGrid: [2, 8],
    caterpillarLines: 22,
    caterpillarSize: 5,
    trackAnchorXMult: 6,
    turretSpeed: TurretSpeedConfig.heavy,
    reloadTime: ReloadConfig.heavy,
    bulletCaliber: BulletCaliber.Heavy,
    bulletOffsetYMult: 13,
    colors: {
        tracks: new Float32Array([0.5, 0.5, 0.5, 1]),
        turret: new Float32Array([0.5, 1, 0.5, 1]),
    },
};

/**
 * Player Tank Configuration
 * Based on Medium tank but with enhanced speed and reload.
 */
export const PlayerTankConfig: TankStats = {
    type: VehicleType.PlayerTank,
    engine: EngineType.v8_turbo,
    size: 6,
    padding: 7,
    density: 300,
    colliderRadius: 80,
    hullSize: [8, 10],
    turretSize: [8, 8],
    hullGrid: [8, 11],
    turretHeadGrid: [6, 7],
    turretGunGrid: [2, 10],
    caterpillarLines: 22,
    caterpillarSize: 3,
    trackAnchorXMult: 5,
    turretSpeed: TurretSpeedConfig.player,
    reloadTime: ReloadConfig.player,
    bulletCaliber: BulletCaliber.Medium,
    bulletOffsetYMult: 13,
    colors: {
        tracks: new Float32Array([0.4, 0.4, 0.5, 1]),
        turret: new Float32Array([0.4, 0.8, 1, 1]),
    },
};

// =============================================================================
// HARVESTER CONFIGURATION
// =============================================================================

export type HarvesterStats = {
    type: VehicleType;
    engine: EngineType;
    size: number;
    padding: number;
    density: number;
    colliderRadius: number;
    hullSize: [number, number];
    barrierSize: [number, number];
    hullGrid: [number, number];
    barrierGrid: [number, number];
    caterpillarLines: number;
    caterpillarSize: number;
    trackAnchorXMult: number;
    turretSpeed: number;
    /** Scoop configuration */
    scoop: {
        sideLength: number;
        frontLength: number;
        offsetYMult: number;
    };
    /** Shield arc configuration */
    shield: {
        partSize: number;
        radiusMult: number;
        arcAngle: number;
        partsCount: number;
        /** Inner arc scale */
        innerScale: number;
    };
    colors: {
        tracks: Float32Array;
        barrier: Float32Array;
        scoop: Float32Array;
        shield: Float32Array;
    };
};

export const HarvesterConfig: HarvesterStats = {
    type: VehicleType.Harvester,
    engine: EngineType.v12,
    size: 6,
    padding: 7,
    density: 350,
    colliderRadius: 85,
    hullSize: [10, 10],
    barrierSize: [10, 6],
    hullGrid: [10, 12],
    barrierGrid: [8, 4],
    caterpillarLines: 20,
    caterpillarSize: 4,
    trackAnchorXMult: 6,
    turretSpeed: TurretSpeedConfig.harvester,
    scoop: {
        sideLength: 6,
        frontLength: 10,
        offsetYMult: 11.5,
    },
    shield: {
        partSize: 8,
        radiusMult: 16, // radius = partSize * radiusMult
        arcAngle: PI * 0.6, // ~108 degrees
        partsCount: 35,
        innerScale: 0.8,
    },
    colors: {
        tracks: new Float32Array([0.4, 0.4, 0.4, 1]),
        barrier: new Float32Array([0.8, 0.6, 0.2, 1]),
        scoop: new Float32Array([0.6, 0.5, 0.3, 1]),
        shield: new Float32Array([0.3, 0.7, 1.0, 0.6]),
    },
};

// =============================================================================
// MELEE CAR CONFIGURATION
// =============================================================================

export type MeleeCarStats = {
    type: VehicleType;
    engine: EngineType;
    size: number;
    padding: number;
    density: number;
    colliderRadius: number;
    hullSize: [number, number];
    hullGrid: [number, number];
    /** Physics damping (less = more sliding) */
    linearDamping: number;
    angularDamping: number;
    /** Wheel configuration */
    wheel: {
        size: number;
        heightMult: number;
        maxSteeringAngle: number;
        steeringSpeed: number;
    };
    colors: {
        wheel: Float32Array;
    };
};

export const MeleeCarConfig: MeleeCarStats = {
    type: VehicleType.MeleeCar,
    engine: EngineType.v8,
    size: 4,
    padding: 5,
    density: 200,
    colliderRadius: 45,
    hullSize: [6, 10],
    hullGrid: [6, 10],
    linearDamping: 3,
    angularDamping: 4,
    wheel: {
        size: 6,
        heightMult: 1.5,
        maxSteeringAngle: PI / 5, // ~36 degrees
        steeringSpeed: PI * 3,
    },
    colors: {
        wheel: new Float32Array([0.2, 0.2, 0.2, 1]),
    },
};

// =============================================================================
// VEHICLE LOOKUP
// =============================================================================

/**
 * Get tank configuration by vehicle type.
 * Returns undefined for non-tank vehicles.
 */
export function getTankConfig(type: VehicleType): TankStats | undefined {
    switch (type) {
        case VehicleType.LightTank:
            return LightTankConfig;
        case VehicleType.MediumTank:
            return MediumTankConfig;
        case VehicleType.HeavyTank:
            return HeavyTankConfig;
        case VehicleType.PlayerTank:
            return PlayerTankConfig;
        default:
            return undefined;
    }
}

