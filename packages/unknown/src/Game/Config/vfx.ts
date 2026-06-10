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
    explosion: { color: [1.0, 0.5, 0.2], intensity: 6.0, duration: 600 },
} as const;

/**
 * Light carried by each stream particle (alpha-0 SDF circle + LightEmitter on
 * the particle entity itself, so the glow follows it). Intensities are a
 * fraction of the standard flash (3.0): flame 30%, frost 10%.
 */
export const StreamParticleLightConfig = {
    flame: { color: [1.0, 0.55, 0.15], intensity: 3.0 * 0.3, radius: 12 },
    frost: { color: [0.45, 0.78, 1.0], intensity: 3.0 * 0.1, radius: 12 },
} as const;

