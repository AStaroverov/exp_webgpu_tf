/**
 * Vehicles Configuration
 *
 * All vehicle types and their characteristics:
 * tanks (light, medium, heavy, player), harvester, melee car.
 */

import { PI } from "../../../../../lib/math.ts";
import { BulletCaliber, StreamCaliber, TurretSpeedConfig } from "./weapons.ts";

/**
 * All available vehicle types in the game.
 */
export enum VehicleType {
  LightTank = 0,
  MediumTank = 1,
  HeavyTank = 2,
  RocketTank = 3, // Heavy chassis mounting a slow-reload rocket launcher
  Harvester = 4, // Bulldozer with barrier and scoop for collecting debris
  MeleeCar = 5, // Fast 4-wheeled car for ramming
  FlameTank = 6, // Medium chassis mounting the flamethrower stream gun
  FrostTank = 7, // Medium chassis mounting the freeze stream gun
  EmpTank = 8, // Medium chassis lobbing the EMP stun grenade
}

/**
 * Engine types determine movement speed and power.
 */
export enum EngineType {
  v6 = 0,
  v8 = 1,
  v12 = 2,
}

export const EngineLabels: Record<EngineType, string> = {
  [EngineType.v6]: "v6",
  [EngineType.v8]: "v8",
  [EngineType.v12]: "v12",
};

/**
 * Global multiplier applied to EVERY engine's movement impulse below.
 * Bump this to speed up (or slow down) all vehicles at once.
 * Module-local: it is baked into {@link EngineConfig}, not exported.
 */
const MovementSpeedMult = 1.5;

/**
 * Engine power multipliers relative to base values.
 * `impulseMult` is already scaled by {@link MovementSpeedMult}.
 */
export const EngineConfig: Record<EngineType, { impulseMult: number; rotationMult: number }> = {
  [EngineType.v6]: {
    impulseMult: 0.8 * MovementSpeedMult,
    rotationMult: 0.9,
  },
  [EngineType.v8]: {
    impulseMult: 1.0 * MovementSpeedMult,
    rotationMult: 1.0,
  },
  [EngineType.v12]: {
    impulseMult: 2.0 * MovementSpeedMult,
    rotationMult: 3.0,
  },
};

/** Chassis/turret stats shared by every tank, armed or not. */
export type VehicleStats = {
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
  /** Hull dimensions [width, height] in padding units */
  hullSize: [number, number];
  /** Turret dimensions [width, height] in padding units */
  turretSize: [number, number];
  /** Hull parts grid [cols, rows] */
  hullGrid: [number, number];
  /** Turret head grid [cols, rows] */
  turretHeadGrid: [number, number];
  /** Number of caterpillar lines */
  caterpillarLines: number;
  /** Caterpillar part size */
  caterpillarSize: number;
  /** Track anchor X offset from center */
  trackAnchorXMult: number;
  /** Turret rotation speed (radians/sec) */
  turretSpeed: number;
  /** Colors */
  colors: {
    tracks: Float32Array;
    turret: Float32Array;
  };
};

/**
 * Gun armament — present ONLY on vehicles that fire rounds. A gunless vehicle
 * omits this group entirely, so "gunless" is the absence of `gun`, not a gun full
 * of zeroed-out values.
 */
export type GunStats = {
  /** Turret gun grid [cols, rows] */
  gunGrid: [number, number];
  /** Bullet caliber (its config carries damage, reload, range) */
  caliber: BulletCaliber;
  /** Bullet start position Y offset multiplier */
  bulletOffsetYMult: number;
};

/** Stream armament — present ONLY on stream-gun vehicles (flame / frost hose). */
export type StreamStats = {
  /** Row of `StreamCaliberConfig` the turret's `StreamFirearms` points at */
  caliber: StreamCaliber;
};

export type TankStats = VehicleStats & { gun?: GunStats; stream?: StreamStats };

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
  hullSize: [8, 10],
  turretSize: [6, 6],
  hullGrid: [8, 10],
  turretHeadGrid: [5, 6],
  caterpillarLines: 12,
  caterpillarSize: 6, // size - 1
  trackAnchorXMult: 4,
  turretSpeed: TurretSpeedConfig.light,
  gun: {
    gunGrid: [2, 6],
    caliber: BulletCaliber.Light,
    bulletOffsetYMult: 9,
  },
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
  density: 275,
  hullSize: [8, 12],
  turretSize: [8, 8],
  hullGrid: [8, 11],
  turretHeadGrid: [6, 7],
  caterpillarLines: 22,
  caterpillarSize: 3,
  trackAnchorXMult: 5,
  turretSpeed: TurretSpeedConfig.medium,
  gun: {
    gunGrid: [2, 10],
    caliber: BulletCaliber.Medium,
    bulletOffsetYMult: 13,
  },
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
  density: 300,
  hullSize: [10, 14],
  turretSize: [8, 8],
  hullGrid: [10, 14],
  turretHeadGrid: [7, 9],
  caterpillarLines: 22,
  caterpillarSize: 5,
  trackAnchorXMult: 6,
  turretSpeed: TurretSpeedConfig.heavy,
  gun: {
    gunGrid: [2, 8],
    caliber: BulletCaliber.Heavy,
    bulletOffsetYMult: 13,
  },
  colors: {
    tracks: new Float32Array([0.5, 0.5, 0.5, 1]),
    turret: new Float32Array([0.5, 1, 0.5, 1]),
  },
};

/**
 * Rocket Tank Configuration
 * Heavy chassis with a slow-reload rocket launcher. Mirrors the Heavy tank
 * stats but swaps the gun to the rocket launcher.
 */
export const RocketTankConfig: TankStats = {
  type: VehicleType.RocketTank,
  engine: EngineType.v6,
  size: 9,
  padding: 9, // size + 1
  density: 450,
  hullSize: [10, 14],
  turretSize: [8, 8],
  hullGrid: [10, 14],
  turretHeadGrid: [7, 9],
  caterpillarLines: 22,
  caterpillarSize: 5,
  trackAnchorXMult: 6,
  turretSpeed: 0,
  gun: {
    gunGrid: [2, 8],
    caliber: BulletCaliber.Rocket,
    bulletOffsetYMult: 13,
  },
  colors: {
    tracks: new Float32Array([0.5, 0.5, 0.5, 1]),
    turret: new Float32Array([0.8, 0.3, 0.2, 1]),
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

/**
 * Stream tanks: the Medium chassis with the gun swapped for a stream weapon
 * (`StreamFirearms` on the turret instead of `Firearms`), parameterized by the
 * stream caliber.
 */
const { gun: _mediumGun, ...MediumChassis } = MediumTankConfig;

export const FlameTankConfig: TankStats = {
  ...MediumChassis,
  type: VehicleType.FlameTank,
  stream: { caliber: StreamCaliber.Flamethrower },
};

export const FrostTankConfig: TankStats = {
  ...MediumChassis,
  type: VehicleType.FrostTank,
  stream: { caliber: StreamCaliber.FreezeGun },
};

/** EMP tank: the Medium chassis with the gun rechambered for the EMP grenade. */
export const EmpTankConfig: TankStats = {
  ...MediumTankConfig,
  type: VehicleType.EmpTank,
  gun: { ...MediumTankConfig.gun!, caliber: BulletCaliber.EmpGrenade },
};

/**
 * Get tank configuration by vehicle type.
 * Returns undefined for non-tank vehicles.
 */
export function getTankConfig(type: VehicleType): TankStats {
  switch (type) {
    case VehicleType.LightTank:
      return LightTankConfig;
    case VehicleType.MediumTank:
      return MediumTankConfig;
    case VehicleType.HeavyTank:
      return HeavyTankConfig;
    case VehicleType.RocketTank:
      return RocketTankConfig;
    case VehicleType.FlameTank:
      return FlameTankConfig;
    case VehicleType.FrostTank:
      return FrostTankConfig;
    case VehicleType.EmpTank:
      return EmpTankConfig;
    default:
      throw new Error(`Unknown vehicle type ${type}`);
  }
}

/** Front light bar: 4 destructible slot parts (SlotPartType.Headlight) along
 *  the hull's front edge that emit light into the RC lighting pass. */
export const HeadlightConfig = {
  color: new Float32Array([1.0, 1.0, 1.0, 1.0]),
  intensity: 1.2,
  directional: true,
} as const;
