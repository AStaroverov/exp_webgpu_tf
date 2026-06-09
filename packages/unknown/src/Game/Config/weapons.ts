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
        width: 8 * 1.6,
        height: 3 * 1.6,
        speed: 450 * 1.6,
        density: 5_000,
        damage: 3 * 3,
        linearDamping: 0.4, // Light bullets lose speed quickly
        maxDistance: HexGridConfig.radius * 2.6,
        health: (3 * 1.6) / 10,
    },
    
    [BulletCaliber.Medium]: {
        width: 10 * 1.6,
        height: 4 * 1.6,
        speed: 525 * 1.6,
        density: 5_000,
        damage: 4 * 3,
        linearDamping: 0.3, // Medium bullets have moderate drag
        maxDistance: HexGridConfig.radius * 4.6,
        health: (4 * 1.6) / 10,
    },
    
    [BulletCaliber.Heavy]: {
        width: 12 * 1.6,
        height: 5 * 1.6,
        speed: 600 * 1.6,
        density: 5_000,
        damage: 6 * 3,
        linearDamping: 0.2, // Heavy bullets maintain speed longer
        maxDistance: HexGridConfig.radius * 6.6,
        health: (5 * 1.6) / 10,
    },

    [BulletCaliber.Rocket]: {
        width: 20 * 1.6,
        height: 8 * 1.6,
        speed: 350 * 1.6, 
        density: 5_000,
        damage: 10,       
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
 * Default turret rotation speeds for different tank classes.
 * Values are in radians per second.
 */
export const TurretSpeedConfig = {
    /** Light tank turret - very fast */
    light: PI * 0.8,
    
    /** Medium tank turret - balanced */
    medium: PI * 0.6,
    
    /** Heavy tank turret - slow but powerful */
    heavy: PI * 0.25,
    
    /** Player tank turret - enhanced speed */
    player: PI * 0.8,
    
    /** Harvester barrier - slow rotation */
    harvester: PI * 0.4,
} as const;

// =============================================================================
// RELOAD TIMES
// =============================================================================

/**
 * Reload durations in milliseconds.
 */
export const ReloadConfig = {
    /** Light tank - fast reload */
    light: 800,
    
    /** Medium tank - balanced reload */
    medium: 1200,
    
    /** Heavy tank - slow reload */
    heavy: 1600,
    
    /** Player tank - enhanced reload */
    player: 300,

    /** Rocket launcher - very long reload */
    rocketLauncher: 5000,
} as const;

export type BulletSpeedType = typeof BulletSpeedConfig;
export type TurretSpeedType = typeof TurretSpeedConfig;
export type ReloadType = typeof ReloadConfig;

