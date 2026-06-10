/**
 * Weapons Configuration
 * 
 * All weapon-related settings: bullet calibers, projectile physics, damage values.
 */

import { PI } from '../../../../../lib/math.ts';
import { HexGridConfig } from '../Map/HexConfig.ts';
import { ExplodableSettings } from '../ECS/Components/Explodable.ts';
import { DamageKind } from '../ECS/Components/Damagable.ts';

// =============================================================================
// BULLET SPEED LIMITS
// =============================================================================

export const BulletSpeedConfig = {
    /** Maximum allowed bullet speed */
    max: 2000,

    /** Minimum speed before bullet is destroyed */
    min: 200,
} as const;

// =============================================================================
// BULLET CALIBERS
// =============================================================================

/**
 * Bullet caliber types - determines projectile characteristics
 */
export enum BulletCaliber {
    Light = 0,
    Medium = 1,
    Heavy = 2,
    Rocket = 3,
}

export type BulletCaliberStats = {
    /** Width of the bullet in pixels */
    width: number;
    /** Height of the bullet in pixels */
    height: number;
    /** Initial speed of the bullet */
    speed: number;
    /** Physics density (affects momentum) */
    density: number;
    /** Damage dealt on hit */
    damage: number;
    /** Reload duration in ms for a gun firing this caliber */
    reloadTime: number;
    /** Linear damping (speed loss over time) */
    linearDamping: number;
    /** Maximum distance the bullet may travel from its spawn point before it is destroyed */
    maxDistance: number;
    /**
     * Hit-points of the projectile itself. A tiny value makes the projectile die
     * (and, for rockets, detonate) on any contact.
     */
    health: number;
    /** When set, the projectile detonates (VFX + area damage) on destruction. */
    explosion?: ExplodableSettings;
};

/**
 * Bullet caliber statistics.
 * Configure all projectile properties here.
 */
export const BulletCaliberConfig: Record<BulletCaliber, BulletCaliberStats> = {
    [BulletCaliber.Light]: {
        width: 8,
        height: 5,
        speed: 650,
        density: 5_000,
        damage: 3 * 3,
        reloadTime: 750, // Light tank - fast reload
        linearDamping: 0.4, // Light bullets lose speed quickly
        maxDistance: HexGridConfig.radius * 4.6,
        health: (3 * 1.6) / 10,
    },
    
    [BulletCaliber.Medium]: {
        width: 10,
        height: 6,
        speed: 800,
        density: 5_000,
        damage: 4 * 3,
        reloadTime: 1500, // Medium tank - balanced reload
        linearDamping: 0.3, // Medium bullets have moderate drag
        maxDistance: HexGridConfig.radius * 6.6,
        health: (4 * 1.6) / 10,
    },
    
    [BulletCaliber.Heavy]: {
        width: 12,
        height: 7,
        speed: 950,
        density: 5_000,
        damage: 6 * 3,
        reloadTime: 3000, // Heavy tank - slow reload
        linearDamping: 0.2, // Heavy bullets maintain speed longer
        maxDistance: HexGridConfig.radius * 8.6,
        health: (5 * 1.6) / 10,
    },

    [BulletCaliber.Rocket]: {
        width: 20,
        height: 12,
        speed: 450, 
        density: 5_000,
        damage: 10,
        reloadTime: 6000, // Rocket launcher - very long reload
        linearDamping: 0.1,
        maxDistance: HexGridConfig.radius * 8,
        health: 0.001,        
        explosion: {
            damage: 5,
            radius: 100,        
            vfxSize: 30 * 6,
            lightRadius: 30 * 8,
        },
    },
} as const;

// =============================================================================
// STREAM WEAPONS (flamethrower / freeze gun)
// =============================================================================

/**
 * Frost-kind damage specialty: every Frost hit on a part deepens the victim
 * vehicle's freeze (`Slowed.slowMul` grows, cap 1); it thaws back per tick.
 */
export const FrostSlowConfig = {
    /** Freeze added per Frost-kind damage event (`slowMul` grows by this) */
    freezePerHit: 0.05,
    /** Freeze thawed per tick (`slowMul` shrinks by this each game tick) */
    thawPerTick: 0.25,
} as const;

export type StreamCaliberStats = {
    /** Particles per emit */
    count: number;
    /** Particle launch speed */
    speed: number;
    /** Half-angle of the spray cone in radians */
    spreadRad: number;
    /** Particle lifetime in ms (DestroyByTimeout) */
    lifetimeMs: number;
    /** Each enemy-part overlap subtracts this from the particle's remaining lifetime (pass-through decay) */
    hitLifeCostMs: number;
    /** Sensor collider radius in world pixels */
    particleRadius: number;
    /** Linear damping so particles decelerate and dissipate like a spray */
    linearDamping: number;
    /** Min ms between emits while the held flag is up (framerate-independent rate) */
    emitIntervalMs: number;
    /** FireStream held-window duration (~1000) */
    holdMs: number;
    /** Fraction of holdMs at which to scheduleRequestNext (~0.8) */
    requestNextFrac: number;
    /** Seeded sinusoidal steering — particles meander like real flame tongues / cold wisps */
    wander: {
        /** Peak turn rate in rad/s (amplitude of the steering sine) */
        angularSpeed: number;
        /** Steering oscillation frequency in Hz */
        frequency: number;
    };
    /** THE divergence field: the damage kind particles deal (Fire / Frost) */
    kind: DamageKind;
    /** Instant damage to the struck part per overlap */
    damage: number;
    /** Damage-over-time stamped on the struck part; duration refreshed per overlap */
    dot: { dps: number; durationMs: number };
    /** Tint applied to struck tank parts while the effect is active */
    tint: [number, number, number];
};

/** Index into `StreamCaliberConfig` — the rows are ordered by this enum. */
export enum StreamCaliber {
    Flamethrower = 0,
    FreezeGun = 1,
}

/**
 * Stream weapon rows — identical in shape; only `kind`, the damage numbers and
 * `tint` differ between the flamethrower and the freeze gun.
 */
export const StreamCaliberConfig: StreamCaliberStats[] = [
    /* Flamethrower */ {
        count: 4,
        speed: 240,
        spreadRad: 0.20,
        lifetimeMs: 800,
        hitLifeCostMs: 20,
        particleRadius: 4,
        linearDamping: 1.2,
        emitIntervalMs: 40,
        holdMs: 1000,
        requestNextFrac: 0.8,
        wander: { angularSpeed: 3.5, frequency: 2.5 },
        kind: DamageKind.Fire,
        damage: 0.05,
        dot: { dps: 0.05, durationMs: 3000 },
        tint: [1.0, 0.45, 0.1],
    },
    /* Freeze gun */ {
        count: 4,
        speed: 240,
        spreadRad: 0.20,
        lifetimeMs: 800,
        hitLifeCostMs: 20,
        particleRadius: 4,
        linearDamping: 1.2,
        emitIntervalMs: 40,
        holdMs: 1000,
        requestNextFrac: 0.8,
        wander: { angularSpeed: 2.5, frequency: 1.8 },
        kind: DamageKind.Frost,
        damage: 0.02,
        dot: { dps: 0.02, durationMs: 5000 },
        tint: [0.3, 0.7, 1.0],
    },
];

// =============================================================================
// TURRET CONFIGURATION
// =============================================================================

/**
 * Global multiplier applied to EVERY turret rotation speed below.
 * Bump this to speed up (or slow down) all turrets at once.
 * Module-local: it is baked into {@link TurretSpeedConfig}, not exported.
 */
const TurretSpeedMult = 2;

/**
 * Turret rotation speeds (radians/sec) per vehicle class — the single source of
 * truth read by entity factories. {@link TurretSpeedMult} is already baked in.
 */
export const TurretSpeedConfig = {
    /** Light tank turret */
    light: PI * 0.3 * TurretSpeedMult,

    /** Medium tank turret */
    medium: PI * 0.2 * TurretSpeedMult,

    /** Heavy tank turret - slow but powerful */
    heavy: PI * 0.1 * TurretSpeedMult,

    /** Harvester barrier - slow rotation for heavy barrier */
    harvester: PI * 0.4 * TurretSpeedMult,
} as const;

export type BulletSpeedType = typeof BulletSpeedConfig;
export type TurretSpeedType = typeof TurretSpeedConfig;

