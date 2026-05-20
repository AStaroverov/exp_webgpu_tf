import { addComponent, removeComponent, EntityId, World, hasComponent } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export enum SoundType {
    None = 0,
    TankMove = 1,
    TankShoot = 2,
    TankHit = 3,
    DebrisCollect = 4,
}

export enum SoundState {
    Stopped = 0,
    Playing = 1,
    Paused = 2,
}

export const createSoundComponent = defineComponent((Sound) => {
    const type = TypedArray.i8(delegate.defaultSize);
    const state = TypedArray.i8(delegate.defaultSize);
    const loop = TypedArray.i8(delegate.defaultSize);
    const volume = TypedArray.f32(delegate.defaultSize);
    const _audioIndex = TypedArray.i16(delegate.defaultSize);

    return {
        type,
        state,
        loop,
        volume,
        _audioIndex,

        addComponent(world: World, eid: EntityId, t: SoundType, options?: {
            loop?: boolean;
            volume?: number;
            autoplay?: boolean;
        }) {
            addComponent(world, eid, Sound);
            type[eid] = t;
            loop[eid] = options?.loop ? 1 : 0;
            volume[eid] = options?.volume ?? 1;
            state[eid] = options?.autoplay ? SoundState.Playing : SoundState.Stopped;
            _audioIndex[eid] = -1;
        },

        removeComponent(world: World, eid: EntityId) {
            removeComponent(world, eid, Sound);
            type[eid] = SoundType.None;
            state[eid] = SoundState.Stopped;
            loop[eid] = 0;
            volume[eid] = 0;
            _audioIndex[eid] = -1;
        },

        play(eid: EntityId) { state[eid] = SoundState.Playing; },
        stop(eid: EntityId) { state[eid] = SoundState.Stopped; },
        pause(eid: EntityId) { state[eid] = SoundState.Paused; },

        setVolume(eid: EntityId, v: number) {
            volume[eid] = Math.max(0, Math.min(1, v));
        },

        isPlaying(eid: EntityId): boolean { return state[eid] === SoundState.Playing; },
        hasSound(world: World, eid: EntityId): boolean { return hasComponent(world, eid, Sound); },
    };
});

export const createDestroyOnSoundFinishComponent = defineComponent((DestroyOnSoundFinish) => ({
    addComponent(world: World, eid: EntityId) { addComponent(world, eid, DestroyOnSoundFinish); },
}));

export const createSoundParentRelativeComponent = defineComponent((SoundParentRelative) => ({
    addComponent(world: World, eid: EntityId) { addComponent(world, eid, SoundParentRelative); },
}));
