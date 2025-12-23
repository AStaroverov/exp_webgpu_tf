/**
 * Weapons Configuration
 * 
 * All weapon-related settings: bullet calibers, projectile physics, damage values.
 */

import { PI } from '../../../../../lib/math.ts';

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
};

/**
 * Bullet caliber statistics.
 * Configure all projectile properties here.
 */
export const BulletCaliberConfig: Record<BulletCaliber, BulletCaliberStats> = {
    [BulletCaliber.Light]: {
        width: 3,
        height: 8,
        speed: 500,
        density: 3_000,
        damage: 3,
        linearDamping: 0.4, // Light bullets lose speed quickly
    },
    
    [BulletCaliber.Medium]: {
        width: 5,
        height: 10,
        speed: 650,
        density: 6_000,
        damage: 6,
        linearDamping: 0.2, // Medium bullets have moderate drag
    },
    
    [BulletCaliber.Heavy]: {
        width: 7,
        height: 14,
        speed: 800,
        density: 10_000,
        damage: 10,
        linearDamping: 0.075, // Heavy bullets maintain speed longer
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
} as const;

export type BulletSpeedType = typeof BulletSpeedConfig;
export type TurretSpeedType = typeof TurretSpeedConfig;
export type ReloadType = typeof ReloadConfig;

