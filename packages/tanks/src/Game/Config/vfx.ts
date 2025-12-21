/**
 * VFX Configuration
 * 
 * Visual effects: explosions, muzzle flashes, particles.
 */

export const ExplosionConfig = {
    /** Default explosion duration (ms) */
    defaultDuration: 1050,
    
    /** Muzzle flash duration (ms) */
    muzzleFlashDuration: 1050,
    
    /** Size multiplier for muzzle flash (relative to bullet width) */
    muzzleFlashSizeMult: 5,
} as const;

export type ExplosionType = typeof ExplosionConfig;

