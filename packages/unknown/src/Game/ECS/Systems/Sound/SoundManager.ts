/**
 * Sound Manager - centralized audio management for the game
 * Uses Web Audio API (AudioContext) for low-latency playback with spatial audio support
 */

import { WebAudioTrack } from "./WebAudioTrack.ts";
import {
  SoundBuses,
  SoundMaster,
  SoundLimiter,
  SoundJitter,
  SoundDuck,
  type SoundBusId,
  type SoundDuckId,
} from "../../../Config/sound.ts";

export interface SoundConfig {
  src: string | string[]; // Single source or array for random selection
  maxInstances?: number; // Max simultaneous plays of this sound
  volume?: number;
  loop?: boolean;
  bus?: SoundBusId; // Which per-type bus to route this sound through
}

export interface PlayOptions {
  volume?: number;
  loop?: boolean;
  // For spatial audio
  x?: number;
  y?: number;
}

interface SoundInstance {
  track: WebAudioTrack;
  inUse: boolean;
  srcIndex: number; // Which source this instance uses
}

interface LoadedSound {
  buffers: AudioBuffer[]; // One buffer per source
  instances: SoundInstance[];
  nextSrcIndex: number; // Round-robin cursor over buffers (per-id)
}

class SoundManager {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private limiter: DynamicsCompressorNode | null = null;
  private busGains: Map<SoundBusId, GainNode> = new Map();

  private sounds: Map<string, LoadedSound> = new Map();
  private configs: Map<string, SoundConfig> = new Map();
  private masterVolume: number = SoundMaster.volume;
  private enabled = true;

  /**
   * Initialize AudioContext (must be called after user interaction).
   * Builds the fixed graph: busGain[type] -> masterGain -> limiter -> destination.
   */
  private ensureContext(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext();

      // Safety limiter (DynamicsCompressor) — last node before destination.
      this.limiter = this.ctx.createDynamicsCompressor();
      this.limiter.threshold.value = SoundLimiter.threshold;
      this.limiter.knee.value = SoundLimiter.knee;
      this.limiter.ratio.value = SoundLimiter.ratio;
      this.limiter.attack.value = SoundLimiter.attack;
      this.limiter.release.value = SoundLimiter.release;
      this.limiter.connect(this.ctx.destination);

      // Single master gain — headroom below clipping.
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = this.masterVolume;
      this.masterGain.connect(this.limiter);

      // Per-type buses at their resting levels.
      for (const id of Object.keys(SoundBuses) as SoundBusId[]) {
        const busGain = this.ctx.createGain();
        busGain.gain.value = SoundBuses[id];
        busGain.connect(this.masterGain);
        this.busGains.set(id, busGain);
      }
    }

    // Resume if suspended (browser autoplay policy). Fire-and-forget here on
    // the hot path; the awaited variant is resume() below.
    if (this.ctx.state === "suspended") {
      this.ctx.resume();
    }

    return this.ctx;
  }

  /**
   * Ensure the context exists and await its resume (browser autoplay policy).
   * Called once from enableSound after the user gesture so playback that
   * follows loadGameSounds is on a running context.
   */
  async resume(): Promise<void> {
    const ctx = this.ensureContext();
    if (ctx.state === "suspended") {
      await ctx.resume();
    }
  }

  /**
   * Preload a sound for later use
   */
  async load(id: string, config: SoundConfig): Promise<void> {
    const ctx = this.ensureContext();
    this.configs.set(id, config);

    const sources = Array.isArray(config.src) ? config.src : [config.src];
    const maxInstances = config.maxInstances ?? 5;

    // Load all audio buffers. A missing/undecodable asset (e.g. a 404 that the
    // dev server answers with index.html, which fails decodeAudioData) must NOT
    // throw — it leaves the sound unregistered so play() returns null, same as
    // an empty variant pool. We warn and skip the bad source.
    const decoded = await Promise.all(
      sources.map(async (src) => {
        try {
          const response = await fetch(src);
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const arrayBuffer = await response.arrayBuffer();
          return await ctx.decodeAudioData(arrayBuffer);
        } catch (err) {
          console.warn(`SoundManager: failed to load "${src}" — sound disabled:`, err);
          return null;
        }
      }),
    );
    const buffers = decoded.filter((b): b is AudioBuffer => b !== null);
    if (buffers.length === 0) {
      // Nothing decoded: config is kept for signature compat, but no playable
      // sound is registered, so play(id) returns null gracefully.
      return;
    }

    // Route through the requested per-type bus; fall back to master if unset.
    const destination =
      (config.bus && this.busGains.get(config.bus)) ?? this.masterGain!;

    // Create WebAudioTrack instances
    const instances: SoundInstance[] = [];
    for (let i = 0; i < maxInstances; i++) {
      const track = new WebAudioTrack(ctx, {
        volume: config.volume ?? 1,
        loop: config.loop ?? false,
      });
      track.connect(destination);

      instances.push({
        track,
        inUse: false,
        srcIndex: i % buffers.length,
      });
    }

    this.sounds.set(id, { buffers, instances, nextSrcIndex: 0 });
  }

  /**
   * Play a sound by id
   */
  play(id: string, options: PlayOptions = {}): WebAudioTrack | null {
    if (!this.enabled) return null;

    this.ensureContext();
    const sound = this.sounds.get(id);
    const config = this.configs.get(id);

    if (!sound || !config) {
      return null;
    }

    const { buffers, instances } = sound;

    // Round-robin sample selection by index (per-id cursor) — no random repeats.
    // No-op while there is a single sample; carries variation once art ships
    // multiple variants per type.
    const srcIndex = sound.nextSrcIndex % buffers.length;
    sound.nextSrcIndex = (sound.nextSrcIndex + 1) % buffers.length;

    // Find an available instance
    const instance = instances.find((i) => !i.inUse);

    if (!instance) {
      // All instances busy - skip
      return null;
    }

    instance.inUse = true;
    instance.srcIndex = srcIndex;

    // Stop previous playback if any
    instance.track.stop();

    // Set buffer and options
    const loop = options.loop ?? config.loop ?? false;
    instance.track.setBuffer(buffers[srcIndex]).setLoop(loop);

    // Volume comes from the caller (distance attenuation lives in SoundSystem,
    // the single spatial model). x/y stay in PlayOptions for signature compat
    // but are no longer used here.
    let volume = options.volume ?? config.volume ?? 1;

    // Per-start variation, one-shots only. Looping sounds (engine) get neither
    // pitch nor gain jitter — random detune on a sustained loop wobbles/howls.
    if (!loop) {
      const { pitch, gainDb } = SoundJitter.oneShot;
      // Pitch: 1 ± pitch (uniform).
      const rate = 1 + (Math.random() * 2 - 1) * pitch;
      instance.track.setPlaybackRate(rate);
      // Gain: ±gainDb dB → linear multiplier 10^(dB/20).
      const db = (Math.random() * 2 - 1) * gainDb;
      volume *= Math.pow(10, db / 20);
    } else {
      // Loop: keep pitch neutral (instances are reused from the pool).
      instance.track.setPlaybackRate(1);
    }

    instance.track.setVolume(volume);
    instance.track.play();

    // onEnded ownership belongs to SoundSystem; it calls releaseInstance() to
    // return this instance to the pool. We do NOT register onEnded here.

    return instance.track;
  }

  /**
   * Return a track's pool instance to the available state. Called by the
   * single onEnded owner (SoundSystem) when playback ends naturally.
   */
  releaseInstance(track: WebAudioTrack): void {
    for (const sound of this.sounds.values()) {
      const instance = sound.instances.find((i) => i.track === track);
      if (instance) {
        instance.inUse = false;
        break;
      }
    }
  }

  /**
   * Stop all instances of a sound
   */
  stop(id: string): void {
    const sound = this.sounds.get(id);
    if (!sound) return;

    for (const instance of sound.instances) {
      instance.track.stop();
      instance.inUse = false;
    }
  }

  /**
   * Pause a specific track instance
   */
  pauseInstance(track: WebAudioTrack): void {
    track.pause();
  }

  /**
   * Resume a specific track instance
   */
  resumeInstance(track: WebAudioTrack): void {
    track.resume();
  }

  /**
   * Stop a specific track instance
   */
  stopInstance(track: WebAudioTrack): void {
    // Loops (engine) fade out to avoid a click on the most frequent stop path
    // (tank halting); one-shots hard-stop. All SoundSystem stop paths
    // (topN drop / Stopped / orphan-cleanup / dispose) route through here.
    if (track.loop) {
      track.fadeOutAndStop();
    } else {
      track.stop();
    }

    // Find and mark as not in use
    for (const sound of this.sounds.values()) {
      const instance = sound.instances.find((i) => i.track === track);
      if (instance) {
        instance.inUse = false;
        break;
      }
    }
  }

  /**
   * Duck a per-type bus: dip its gain to `amount` for `holdMs`, then recover to
   * its resting level. Used as an event-driven sidechain (e.g. the engine bus
   * dips on an explosion). Distinct ramp primitives per the Web Audio notes:
   *   - down: linearRampToValueAtTime — fast, deterministically reaches `amount`
   *     (setTargetAtTime would only get ~63% of the dip over downMs).
   *   - recovery: setTargetAtTime(resting, t, tau) — exponential approach; never
   *     an exact deadline, so overlapping ducks compound off the current level.
   * Anchored with cancelScheduledValues + setValueAtTime so no two events land
   * on the param at the same currentTime.
   */
  duck(busId: SoundDuckId, amount: number, holdMs: number): void {
    const ctx = this.ctx;
    const bus = this.busGains.get(busId);
    if (!ctx || !bus) return;

    const now = ctx.currentTime;
    const resting = SoundBuses[busId];
    // Envelope timings key off the bus being ducked. The param is SoundDuckId,
    // so only a bus that actually has a duck config can be passed (compile-time).
    const { downMs, recoveryTau } = SoundDuck[busId];

    const downAt = now + downMs / 1000;
    const recoverAt = downAt + holdMs / 1000;

    const gain = bus.gain;
    // Anchor on the real current value, wiping any in-flight ramps so the dip
    // starts cleanly from wherever the bus is right now.
    gain.cancelScheduledValues(now);
    gain.setValueAtTime(gain.value, now);
    gain.linearRampToValueAtTime(amount, downAt);
    // Hold at the ducked level until recovery begins — only when holdMs > 0, so
    // we never queue a second event at the same currentTime as the ramp endpoint.
    if (recoverAt > downAt) {
      gain.setValueAtTime(amount, recoverAt);
    }
    // Exponential recovery back to the resting level.
    gain.setTargetAtTime(resting, recoverAt, recoveryTau);
  }

  setMasterVolume(volume: number): void {
    this.masterVolume = Math.max(0, Math.min(1, volume));
    if (this.masterGain) {
      this.masterGain.gain.value = this.masterVolume;
    }
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      this.stopAll();
    }
  }

  stopAll(): void {
    for (const [id] of this.sounds) {
      this.stop(id);
    }
  }

  dispose(): void {
    this.stopAll();

    // Dispose all tracks
    for (const sound of this.sounds.values()) {
      for (const instance of sound.instances) {
        instance.track.dispose();
      }
    }

    this.sounds.clear();
    this.configs.clear();

    if (this.ctx) {
      this.ctx.close();
      this.ctx = null;
      this.masterGain = null;
      this.limiter = null;
      this.busGains.clear();
    }
  }
}

export const soundManager = new SoundManager();
