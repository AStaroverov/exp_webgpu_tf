import { query, hasComponent } from "bitecs";
import type { EntityId } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { SoundType, SoundState } from "../../Components/Sound.ts";
import { CameraState } from "../Camera/CameraSystem.ts";
import { soundManager } from "./SoundManager.ts";
import { WebAudioTrack } from "./WebAudioTrack.ts";
import {
  GlobalTransform,
  getMatrixTranslationX,
  getMatrixTranslationY,
} from "../../../../../../renderer/src/ECS/Components/Transform.ts";
import { hypot } from "../../../../../../../lib/math.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import { SoundVariants, SoundDuck, type SoundBusId } from "../../../Config/sound.ts";

const SOUND_IDS: Record<SoundType, string> = {
  [SoundType.None]: "",
  [SoundType.TankMove]: "tank_move",
  [SoundType.TankShoot]: "tank_shoot",
  [SoundType.TankHit]: "tank_hit",
  [SoundType.Explosion]: "explosion",
  [SoundType.Stream]: "stream",
  [SoundType.Emp]: "emp",
};

// Per-type bus routing — set at load time (tracks are pre-connected to their bus).
const SOUND_BUSES: Partial<Record<SoundType, SoundBusId>> = {
  [SoundType.TankMove]: "engine",
  [SoundType.TankShoot]: "shoot",
  [SoundType.TankHit]: "hit",
  [SoundType.Explosion]: "explosion",
  // Stream hose is weapon fire — share the shoot bus.
  [SoundType.Stream]: "shoot",
  // EMP burst is an explosion-class event.
  [SoundType.Emp]: "explosion",
};

// Per-type voice-stealing policy applied when more entities want to play than the
// cap allows. The selection metric is the same (distance-to-camera, nearest wins);
// the modes differ only in whether a NEW voice may displace an already-active one.
//   PreventNew    — keep active voices; new ones only fill spare slots, never evict.
//   StopQuietest  — purely nearest-N; a closer new voice may evict a farther active one.
//   StopFarthest  — nearest-N (alias of StopQuietest while volume is distance-derived).
const enum Resolution {
  PreventNew,
  StopQuietest,
  StopFarthest,
}

const CONFIG = {
  maxSoundsPerType: {
    [SoundType.None]: 0,
    [SoundType.TankMove]: 5,
    [SoundType.TankShoot]: 6,
    [SoundType.TankHit]: 4,
    [SoundType.Explosion]: 4,
    [SoundType.Stream]: 3,
    [SoundType.Emp]: 3,
  },
  resolution: {
    [SoundType.None]: Resolution.StopFarthest,
    [SoundType.TankMove]: Resolution.StopFarthest,
    [SoundType.TankShoot]: Resolution.PreventNew,
    [SoundType.TankHit]: Resolution.StopQuietest,
    [SoundType.Explosion]: Resolution.StopQuietest,
    // Sustained loop: nearest-N, like the engine.
    [SoundType.Stream]: Resolution.StopFarthest,
    [SoundType.Emp]: Resolution.StopQuietest,
  } as Record<SoundType, Resolution>,
  hearingRange: 1000,
  refDistance: 100,
  baseVolume: {
    [SoundType.None]: 0,
    [SoundType.TankMove]: 0.1,
    [SoundType.TankShoot]: 0.2,
    [SoundType.TankHit]: 0.3,
    [SoundType.Explosion]: 0.5,
    [SoundType.Stream]: 0.4,
    [SoundType.Emp]: 0.5,
  },
};

const activeAudios: Map<EntityId, WebAudioTrack> = new Map();

export async function loadGameSounds(): Promise<void> {
  // SoundVariants is the single source of sample pools; each type loads only
  // when its pool is non-empty (an empty pool → no sound registered → play()
  // returns null gracefully, no fetch of a missing asset).
  const LOADABLE: Array<{ type: SoundType; loop: boolean }> = [
    { type: SoundType.TankMove, loop: true },
    { type: SoundType.TankShoot, loop: false },
    { type: SoundType.TankHit, loop: false },
    { type: SoundType.Explosion, loop: false },
    { type: SoundType.Stream, loop: true },
    { type: SoundType.Emp, loop: false },
  ];

  const loads = LOADABLE.flatMap(({ type, loop }) => {
    const src = SoundVariants[SOUND_IDS[type]];
    if (!src || src.length === 0) return [];
    // The looping engine keeps one spare instance over its cap for handoff.
    const maxInstances = CONFIG.maxSoundsPerType[type] + (loop ? 1 : 0);
    return [
      soundManager.load(SOUND_IDS[type], {
        src: [...src],
        maxInstances,
        volume: CONFIG.baseVolume[type],
        loop,
        bus: SOUND_BUSES[type],
      }),
    ];
  });

  await Promise.all(loads);
}

function getEntityPosition(eid: EntityId, { world } = GameDI): { x: number; y: number } {
  const { Parent } = getGameComponents(world);
  if (hasComponent(world, eid, Parent) && hasComponent(world, Parent.id.get(eid), GlobalTransform)) {
    const matrix = GlobalTransform.matrix.getBatch(Parent.id.get(eid));
    return {
      x: getMatrixTranslationX(matrix),
      y: getMatrixTranslationY(matrix),
    };
  }

  if (hasComponent(world, eid, GlobalTransform)) {
    const matrix = GlobalTransform.matrix.getBatch(eid);
    return {
      x: getMatrixTranslationX(matrix),
      y: getMatrixTranslationY(matrix),
    };
  }

  return { x: 0, y: 0 };
}

function getDistanceToCamera(eid: EntityId): number {
  const pos = getEntityPosition(eid);
  const dx = pos.x - CameraState.x;
  const dy = pos.y - CameraState.y;
  return hypot(dx, dy);
}

function calculateDistanceVolume(distance: number, baseVolume: number): number {
  if (distance <= CONFIG.refDistance) return baseVolume;
  if (distance >= CONFIG.hearingRange) return 0;

  const normalized = (distance - CONFIG.refDistance) / (CONFIG.hearingRange - CONFIG.refDistance);
  return baseVolume * (1 - normalized ** 3);
}

export function createSoundSystem({ world } = GameDI) {
  const { Sound } = getGameComponents(world);

  const soundsByType: Map<SoundType, Set<EntityId>> = new Map();
  for (const type of Object.values(SoundType)) {
    if (typeof type === "number") {
      soundsByType.set(type, new Set());
    }
  }

  return function updateSounds(_delta: number): void {
    const soundEids = query(world, [Sound]);

    // Fire the engine-duck at most once per frame, on the first Explosion voice
    // that actually starts this frame (overlapping explosions share one dip).
    let duckedThisFrame = false;

    const entitiesByType: Map<
      SoundType,
      Array<{
        eid: EntityId;
        distance: number;
        wantsToPlay: boolean;
      }>
    > = new Map();

    for (const type of Object.values(SoundType)) {
      if (typeof type === "number" && type !== SoundType.None) {
        entitiesByType.set(type, []);
      }
    }

    for (const eid of soundEids) {
      const type = Sound.type.get(eid) as SoundType;
      if (type === SoundType.None) continue;

      const distance = getDistanceToCamera(eid);
      const wantsToPlay = Sound.state.get(eid) === SoundState.Playing;

      entitiesByType.get(type)?.push({ eid, distance, wantsToPlay });
    }

    for (const [type, entities] of entitiesByType) {
      const baseVolume = CONFIG.baseVolume[type];
      const maxSounds = CONFIG.maxSoundsPerType[type];
      const soundId = SOUND_IDS[type];
      const typeSet = soundsByType.get(type)!;

      const playableEntities = entities
        .filter((e) => e.wantsToPlay && e.distance < CONFIG.hearingRange)
        .sort((a, b) => a.distance - b.distance);

      // Per-type voice stealing. StopQuietest/StopFarthest are plain nearest-N (a
      // closer voice may displace a farther active one). PreventNew prioritizes
      // already-active voices so a new sound never evicts a playing one — it only
      // claims a spare slot.
      let topN: typeof playableEntities;
      if (CONFIG.resolution[type] === Resolution.PreventNew) {
        const active: typeof playableEntities = [];
        const fresh: typeof playableEntities = [];
        for (const e of playableEntities) {
          (typeSet.has(e.eid) ? active : fresh).push(e);
        }
        topN = active.concat(fresh).slice(0, maxSounds);
      } else {
        topN = playableEntities.slice(0, maxSounds);
      }

      const toPlay = new Set(topN.map((e) => e.eid));

      for (const eid of typeSet) {
        if (toPlay.has(eid)) continue;
        const track = activeAudios.get(eid);
        track && soundManager.stopInstance(track);
        handleStoppedSounds(eid);
        activeAudios.delete(eid);
        typeSet.delete(eid);
      }

      for (const { eid, distance } of topN) {
        const volume = calculateDistanceVolume(distance, baseVolume);

        let track = activeAudios.get(eid);

        if (!track) {
          const pos = getEntityPosition(eid);
          const loop = Sound.loop.get(eid) === 1;

          const newTrack = soundManager.play(soundId, {
            volume: volume * Sound.volume.get(eid),
            loop,
            x: pos.x,
            y: pos.y,
          });
          if (newTrack) {
            activeAudios.set(eid, newTrack);
            typeSet.add(eid);

            // First explosion voice this frame ducks the engine bus. Recovery
            // begins right after the dip floor is reached (no extra hold) and is
            // governed by recoveryTau inside duck().
            if (type === SoundType.Explosion && !duckedThisFrame) {
              duckedThisFrame = true;
              soundManager.duck("engine", SoundDuck.engine.amount, 0);
            }

            // SoundSystem is the single onEnded owner: return the pool instance
            // and run the destroy-on-finish policy.
            newTrack.onEnded(() => {
              soundManager.releaseInstance(newTrack);
              handleStoppedSounds(eid);
            });
          }
        } else {
          // Per-frame tracking of a moving source — smooth exponential ramp
          // (setTargetAtTime), never a hard set (zipper noise). This is the
          // >=2nd-frame path, so it cannot collide with the start attack ramp.
          track.rampVolume(volume * Sound.volume.get(eid));
        }
      }
    }

    for (const eid of soundEids) {
      const state = Sound.state.get(eid);
      const track = activeAudios.get(eid);

      if (track) {
        if (state === SoundState.Stopped) {
          soundManager.stopInstance(track);
          activeAudios.delete(eid);
          const type = Sound.type.get(eid) as SoundType;
          soundsByType.get(type)?.delete(eid);
          handleStoppedSounds(eid);
        } else if (state === SoundState.Paused && track.state === "playing") {
          soundManager.pauseInstance(track);
        } else if (state === SoundState.Playing && track.state === "paused") {
          soundManager.resumeInstance(track);
        }
      }
    }

    for (const [eid, track] of activeAudios) {
      if (soundEids.includes(eid)) continue;
      soundManager.stopInstance(track);
      activeAudios.delete(eid);
      for (const typeSet of soundsByType.values()) {
        typeSet.delete(eid);
      }
    }
  };
}

function handleStoppedSounds(eid: EntityId, { world } = GameDI): void {
  const { Destroy, DestroyOnSoundFinish } = getGameComponents(world);
  if (hasComponent(world, eid, DestroyOnSoundFinish) && !hasComponent(world, eid, Destroy)) {
    Destroy.addComponent(world, eid);
  }
}

export function disposeSoundSystem(): void {
  for (const [, track] of activeAudios) {
    soundManager.stopInstance(track);
  }
  activeAudios.clear();
}
