import { addComponent, removeComponent, EntityId, World, hasComponent } from 'bitecs';
import { delegate } from 'renderer/src/delegate.ts';
import { TypedArray } from 'renderer/src/utils.ts';
import { component } from 'renderer/src/ECS/utils.ts';

// Sound type enum - matches loaded sound IDs
export enum SoundType {
    None = 0,
    TankMove = 1,
    TankShoot = 2,
    TankHit = 3,
    DebrisCollect = 4,
}

// Sound state enum
export enum SoundState {
    Stopped = 0,
    Playing = 1,
    Paused = 2,
}

export const Sound = component({
    // Sound type (from SoundType enum)
    type: TypedArray.i8(delegate.defaultSize),
    // Current state (from SoundState enum)
    state: TypedArray.i8(delegate.defaultSize),
    // Whether sound should loop
    loop: TypedArray.i8(delegate.defaultSize),
    // Volume (0-1)
    volume: TypedArray.f32(delegate.defaultSize),
    // Internal: audio instance index (managed by SoundSystem)
    _audioIndex: TypedArray.i16(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, type: SoundType, options?: {
        loop?: boolean;
        volume?: number;
        autoplay?: boolean;
    }): void {
        addComponent(world, eid, Sound);
        Sound.type[eid] = type;
        Sound.loop[eid] = options?.loop ? 1 : 0;
        Sound.volume[eid] = options?.volume ?? 1;
        Sound.state[eid] = options?.autoplay ? SoundState.Playing : SoundState.Stopped;
        Sound._audioIndex[eid] = -1;
    },

    removeComponent(world: World, eid: EntityId): void {
        removeComponent(world, eid, Sound);
        Sound.type[eid] = SoundType.None;
        Sound.state[eid] = SoundState.Stopped;
        Sound.loop[eid] = 0;
        Sound.volume[eid] = 0;
        Sound._audioIndex[eid] = -1;
    },

    play(eid: EntityId): void {
        Sound.state[eid] = SoundState.Playing;
    },

    stop(eid: EntityId): void {
        Sound.state[eid] = SoundState.Stopped;
    },

    pause(eid: EntityId): void {
        Sound.state[eid] = SoundState.Paused;
    },

    setVolume(eid: EntityId, volume: number): void {
        Sound.volume[eid] = Math.max(0, Math.min(1, volume));
    },

    isPlaying(eid: EntityId): boolean {
        return Sound.state[eid] === SoundState.Playing;
    },

    hasSound(world: World, eid: EntityId): boolean {
        return hasComponent(world, eid, Sound);
    },
});

/**
 * Marker component: sound should be destroyed when playback finishes
 */
export const DestroyOnSoundFinish = component({
    addComponent(world: World, eid: EntityId): void {
        addComponent(world, eid, DestroyOnSoundFinish);
    },
});

/**
 * Marker component: sound state depends on parent entity
 * - Position is taken from Parent (via Parent component)
 * - Sound starts/stops based on parent's state (alive/destroyed)
 */
export const SoundParentRelative = component({
    addComponent(world: World, eid: EntityId): void {
        addComponent(world, eid, SoundParentRelative);
    },
});
