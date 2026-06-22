/**
 * VFX Configuration
 *
 * Visual effects: explosions, muzzle flashes, particles.
 */

import { VFXType, VFXTypeValue } from "../ECS/Components/VFX.ts";
import { SoundType } from "../ECS/Components/Sound.ts";

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
  hit: { color: [1.0, 0.55, 0.25], intensity: 3.0, duration: 300 },
  muzzle: { color: [1.0, 0.75, 0.4], intensity: 3.0, duration: 300 },
  explosion: { color: [1.0, 0.5, 0.2], intensity: 6.0, duration: 1050 },
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

export const EmpVfxConfig = {
  tint: [0.65, 0.8, 1.0] as [number, number, number],
  flash: { color: [0.55, 0.8, 1.0], intensity: 4.0, duration: 450 }, // FlashLightConfig row shape
  overlay: { color: [0.55, 0.8, 1.0], lightIntensity: 1.4, lightRadiusPx: 60, radiusPx: 60 },
  // Burst world radius lives on the EmpGrenade caliber row (`explosion.vfxSize`).
  explosion: { durationMs: 450 },
} as const;

/**
 * Detonation visuals row key, stored on `Explodable`. A separate key (not
 * `DamageKind`): bullets and rockets are both Physical but detonate with
 * different visuals.
 */
export enum ExplosionVisual {
  /** Small short impact flash (plain bullets). */
  HitFlash = 0,
  /** Full explosion sprite + bright flash (rockets). */
  Explosion = 1,
  /** EMP burst. */
  Emp = 2,
}

/** Detonation visuals keyed by `ExplosionVisual`. Data lookup, not a type branch. */
export const ExplosionVisualConfig: Record<
  ExplosionVisual,
  {
    vfxType: VFXTypeValue;
    durationMs: number;
    flash: { color: readonly [number, number, number]; intensity: number; duration: number };
    /** Detonation sound; `SoundType.None` = silent (the hit sound covers HitFlash). */
    soundType: SoundType;
  }
> = {
  [ExplosionVisual.HitFlash]: {
    flash: FlashLightConfig.hit,
    vfxType: VFXType.HitFlash,
    durationMs: 400,
    soundType: SoundType.None,
  },
  [ExplosionVisual.Explosion]: {
    flash: FlashLightConfig.explosion,
    vfxType: VFXType.Explosion,
    durationMs: ExplosionConfig.defaultDuration,
    soundType: SoundType.Explosion,
  },
  [ExplosionVisual.Emp]: {
    flash: EmpVfxConfig.flash,
    vfxType: VFXType.EmpExplosion,
    durationMs: EmpVfxConfig.explosion.durationMs,
    soundType: SoundType.Emp,
  },
};
