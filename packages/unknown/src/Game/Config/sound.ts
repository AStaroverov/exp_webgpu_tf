/**
 * Sound Configuration
 *
 * Audio settings for game sounds and effects.
 */

export const SoundConfig = {
  /** Base volume for tank shooting */
  shootBaseVolume: 0.4,

  /** Additional volume per bullet width unit */
  shootVolumePerWidth: 0.085,
} as const;

export type SoundConfigShape = typeof SoundConfig;

/**
 * Per-type bus resting levels (linear gain). One GainNode per bus sits between
 * the per-voice track gain and the master. Ducking envelopes (later phase) live
 * on these nodes; for now they hold a static resting level.
 */
export const SoundBuses = {
  engine: 0.9,
  shoot: 0.8,
  hit: 0.8,
  explosion: 1.0,
} as const;

export type SoundBusId = keyof typeof SoundBuses;

/** Single master gain — headroom so the summed buses do not clip. */
export const SoundMaster = { volume: 0.7 } as const;

/**
 * DynamicsCompressor settings — used ONLY as a safety limiter against clipping
 * (the node has no makeup gain by design; it does not raise quiet parts).
 */
export const SoundLimiter = {
  threshold: -3, // dB
  knee: 0,
  ratio: 20,
  attack: 0.003,
  release: 0.25,
} as const;

/**
 * Ducking envelopes per bus. When a loud transient (e.g. an explosion) plays,
 * the named bus dips to `amount` over `downMs` (linear ramp — deterministically
 * reaches the target) and recovers via `setTargetAtTime(resting, t, recoveryTau)`
 * (exponential approach; ~95% by ~3·tau, never an exact deadline — overlapping
 * ducks compound correctly off the partially-recovered level).
 */
export const SoundDuck = {
  engine: { amount: 0.4, downMs: 30, recoveryTau: 0.13 },
} as const;

export type SoundDuckId = keyof typeof SoundDuck;

/**
 * Per-start variation, applied ONLY to one-shots (`!loop`). Looping sounds
 * (the engine) get NO pitch jitter — random detune on a sustained loop sounds
 * like a wobble/howl. Kills the "machine-gun" artifact of an identical sample
 * repeating dozens of times a second.
 */
export const SoundJitter = {
  oneShot: {
    pitch: 0.05, // ±5% playbackRate
    gainDb: 2, // ±2 dB gain
  },
} as const;

/**
 * Sample variant pools per sound id (round-robin by index). While art ships a
 * single sample per type the round-robin is a no-op and jitter carries the
 * variation; this becomes audible once several samples exist per type.
 */
export const SoundVariants: Record<string, readonly string[]> = {
  tank_move: ["/assets/sounds/tanks/move/engine1.webm"],
  tank_shoot: ["/assets/sounds/tanks/shot/shot.webm"],
  // Two hit samples — round-robin (by index) alternates them so repeated hits
  // do not machine-gun the identical sample.
  tank_hit: ["/assets/sounds/tanks/hit/hit1.webm", "/assets/sounds/tanks/hit/hit2.webm"],
  explosion: ["/assets/sounds/tanks/explosion/explosion1.webm"],
  // Stream-weapon hose — one sustained loop shared by all stream calibers.
  stream: ["/assets/sounds/tanks/shot/thrower.webm"],
  // EMP burst detonation.
  emp: ["/assets/sounds/tanks/shot/emp.webm"],
} as const;
