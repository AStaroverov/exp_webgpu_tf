/**
 * Weapons Configuration
 * 
 * All weapon-related settings: bullet calibers, projectile physics, damage values.
 */

import { PI } from '../../../../../lib/math.ts';
import { HexGridConfig } from '../Map/HexConfig.ts';
import { ExplodableSettings } from '../ECS/Components/Explodable.ts';

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
        height: 3,
        speed: 650,
        density: 5_000,
        damage: 3 * 3,
        reloadTime: 750, // Light tank - fast reload
        linearDamping: 0.4, // Light bullets lose speed quickly
        maxDistance: HexGridConfig.radius * 2.6,
        health: (3 * 1.6) / 10,
    },
    
    [BulletCaliber.Medium]: {
        width: 10,
        height: 4,
        speed: 800,
        density: 5_000,
        damage: 4 * 3,
        reloadTime: 1500, // Medium tank - balanced reload
        linearDamping: 0.3, // Medium bullets have moderate drag
        maxDistance: HexGridConfig.radius * 4.6,
        health: (4 * 1.6) / 10,
    },
    
    [BulletCaliber.Heavy]: {
        width: 12,
        height: 5,
        speed: 950,
        density: 5_000,
        damage: 6 * 3,
        reloadTime: 3000, // Heavy tank - slow reload
        linearDamping: 0.2, // Heavy bullets maintain speed longer
        maxDistance: HexGridConfig.radius * 6.6,
        health: (5 * 1.6) / 10,
    },

    [BulletCaliber.Rocket]: {
        width: 20,
        height: 8,
        speed: 450, 
        density: 5_000,
        damage: 10,
        reloadTime: 5000, // Rocket launcher - very long reload
        linearDamping: 0.1,
        maxDistance: HexGridConfig.radius * 8,
        health: 0.001,         
        explosion: {
            damage: 10,         
            radius: 100,        
            vfxSize: 30 * 6,
            lightRadius: 30 * 8,
        },
    },
} as const;

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

    /** Rocket tank - launcher is bolted to the hull, no turret rotation */
    rocket: 0,

    /** Harvester barrier - slow rotation for heavy barrier */
    harvester: PI * 0.4 * TurretSpeedMult,
} as const;

export type BulletSpeedType = typeof BulletSpeedConfig;
export type TurretSpeedType = typeof TurretSpeedConfig;

