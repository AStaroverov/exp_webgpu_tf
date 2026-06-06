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

/** Light emitted by VFX flashes (emit-only SDF circle, see Entities/LightFlash.ts). */
export const FlashLightConfig = {
    muzzle: { color: [1.0, 0.75, 0.4], intensity: 3.0, duration: 300 },
    hit: { color: [1.0, 0.55, 0.25], intensity: 3.0, duration: 300 },
} as const;

