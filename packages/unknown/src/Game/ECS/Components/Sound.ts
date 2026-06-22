import { addComponent, removeComponent, hasComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export enum SoundType {
  None = 0,
  TankMove = 1,
  TankShoot = 2,
  TankHit = 3,
  Explosion = 5,
  // Stream-weapon hose — one sustained loop shared by every stream caliber
  // (flame / frost), toggled by the firing state (see createStreamFirearmsSystem).
  Stream = 6,
  // EMP burst detonation (createExplodeSystem, driven by ExplosionVisualConfig).
  Emp = 7,
}

export enum SoundState {
  Stopped = 0,
  Playing = 1,
  Paused = 2,
}

export const createSoundComponent = defineComponent((Sound, ctx) => {
  const type = ctx.table.flat(Int8Array);
  const state = ctx.table.flat(Int8Array);
  const loop = ctx.table.flat(Int8Array);
  const volume = ctx.table.flat(Float32Array);

  return {
    type,
    state,
    loop,
    volume,

    addComponent(
      world: World,
      eid: EntityId,
      t: SoundType,
      options?: {
        loop?: boolean;
        volume?: number;
        autoplay?: boolean;
      },
    ) {
      addComponent(world, eid, Sound);
      type.set(eid, t);
      loop.set(eid, options?.loop ? 1 : 0);
      volume.set(eid, options?.volume ?? 1);
      state.set(eid, options?.autoplay ? SoundState.Playing : SoundState.Stopped);
    },

    removeComponent(world: World, eid: EntityId) {
      removeComponent(world, eid, Sound);
    },

    play(eid: EntityId) {
      state.set(eid, SoundState.Playing);
    },
    stop(eid: EntityId) {
      state.set(eid, SoundState.Stopped);
    },
    pause(eid: EntityId) {
      state.set(eid, SoundState.Paused);
    },

    setVolume(eid: EntityId, v: number) {
      volume.set(eid, Math.max(0, Math.min(1, v)));
    },

    isPlaying(eid: EntityId): boolean {
      return state.get(eid) === SoundState.Playing;
    },
    hasSound(world: World, eid: EntityId): boolean {
      return hasComponent(world, eid, Sound);
    },
  };
});

export const createDestroyOnSoundFinishComponent = defineComponent((DestroyOnSoundFinish) => ({
  addComponent(world: World, eid: EntityId) {
    addComponent(world, eid, DestroyOnSoundFinish);
  },
}));

export const createSoundParentRelativeComponent = defineComponent((SoundParentRelative) => ({
  addComponent(world: World, eid: EntityId) {
    addComponent(world, eid, SoundParentRelative);
  },
}));
