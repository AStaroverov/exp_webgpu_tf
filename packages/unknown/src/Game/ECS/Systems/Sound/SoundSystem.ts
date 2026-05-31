import { query, EntityId, hasComponent, removeEntity } from 'bitecs';
import { SoundType, SoundState } from '../../Components/Sound.ts';
import { CameraState } from '../Camera/CameraSystem.ts';
import { soundManager } from './SoundManager.ts';
import { WebAudioTrack } from './WebAudioTrack.ts';
import { GlobalTransform, getMatrixTranslationX, getMatrixTranslationY } from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { hypot } from '../../../../../../../lib/math.ts';
import { getRenderWorldComponents, RenderGameWorld } from '../../createRenderWorld.ts';
import { Worlds } from '../../../DI/Worlds.ts';

const SOUND_IDS: Record<SoundType, string> = {
    [SoundType.None]: '',
    [SoundType.TankMove]: 'tank_move',
    [SoundType.TankShoot]: 'tank_shoot',
    [SoundType.TankHit]: 'tank_hit',
    [SoundType.DebrisCollect]: 'debris_collect',
};

const CONFIG = {
    maxSoundsPerType: {
        [SoundType.None]: 0,
        [SoundType.TankMove]: 5,
        [SoundType.TankShoot]: 8,
        [SoundType.TankHit]: 8,
        [SoundType.DebrisCollect]: 3,
    },
    hearingRange: 1000,
    refDistance: 100,
    baseVolume: {
        [SoundType.None]: 0,
        [SoundType.TankMove]: 0.1,
        [SoundType.TankShoot]: 0.2,
        [SoundType.TankHit]: 0.3,
        [SoundType.DebrisCollect]: 0.3,
    },
};

const activeAudios: Map<EntityId, WebAudioTrack> = new Map();

export async function loadGameSounds(): Promise<void> {
    await Promise.all([
        soundManager.load(SOUND_IDS[SoundType.TankMove], {
            src: '/assets/sounds/tanks/move/engine1.webm',
            maxInstances: CONFIG.maxSoundsPerType[SoundType.TankMove] + 1,
            volume: CONFIG.baseVolume[SoundType.TankMove],
            loop: true,
        }),
        soundManager.load(SOUND_IDS[SoundType.TankShoot], {
            src: [
                '/assets/sounds/tanks/shot/shot.webm',
            ],
            maxInstances: CONFIG.maxSoundsPerType[SoundType.TankShoot],
            volume: CONFIG.baseVolume[SoundType.TankShoot],
            loop: false,
        }),
        soundManager.load(SOUND_IDS[SoundType.TankHit], {
            src: [
                '/assets/sounds/tanks/hit/hit1.webm',
            ],
            maxInstances: CONFIG.maxSoundsPerType[SoundType.TankHit],
            volume: CONFIG.baseVolume[SoundType.TankHit],
            loop: false,
        }),
    ]);
}

function getEntityPosition(world: RenderGameWorld, eid: EntityId): { x: number; y: number } {
    const { Parent } = getRenderWorldComponents(world);
    if (hasComponent(world, eid, Parent) && hasComponent(world, Parent.id[eid], GlobalTransform)) {
        const matrix = GlobalTransform.matrix.getBatch(Parent.id[eid]);
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

function getDistanceToCamera(world: RenderGameWorld, eid: EntityId): number {
    const pos = getEntityPosition(world, eid);
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

export function createSoundSystem({ renderWorld: world } = Worlds) {
    const { Sound } = getRenderWorldComponents(world);

    const soundsByType: Map<SoundType, Set<EntityId>> = new Map();
    for (const type of Object.values(SoundType)) {
        if (typeof type === 'number') {
            soundsByType.set(type, new Set());
        }
    }

    return function updateSounds(_delta: number): void {
        soundManager.setListenerPosition(CameraState.x, CameraState.y);

        const soundEids = query(world, [Sound]);

        const entitiesByType: Map<SoundType, Array<{
            eid: EntityId;
            distance: number;
            wantsToPlay: boolean;
        }>> = new Map();

        for (const type of Object.values(SoundType)) {
            if (typeof type === 'number' && type !== SoundType.None) {
                entitiesByType.set(type, []);
            }
        }

        for (const eid of soundEids) {
            const type = Sound.type[eid] as SoundType;
            if (type === SoundType.None) continue;

            const distance = getDistanceToCamera(world, eid);
            const wantsToPlay = Sound.state[eid] === SoundState.Playing;

            entitiesByType.get(type)?.push({ eid, distance, wantsToPlay });
        }

        for (const [type, entities] of entitiesByType) {
            const baseVolume = CONFIG.baseVolume[type];
            const maxSounds = CONFIG.maxSoundsPerType[type];
            const soundId = SOUND_IDS[type];
            const typeSet = soundsByType.get(type)!;

            const playableEntities = entities
                .filter(e => e.wantsToPlay && (e.distance < CONFIG.hearingRange))
                .sort((a, b) => a.distance - b.distance);

            const topN = playableEntities.slice(0, maxSounds);
            const toPlay = new Set(topN.map(e => e.eid));

            for (const eid of typeSet) {
                if (toPlay.has(eid)) continue;
                const track = activeAudios.get(eid);
                track && soundManager.stopInstance(track);
                handleStoppedSounds(world, eid);
                activeAudios.delete(eid);
                typeSet.delete(eid);
            }

            for (const { eid, distance } of topN) {
                const volume = calculateDistanceVolume(distance, baseVolume);

                let track = activeAudios.get(eid);

                if (!track) {
                    const pos = getEntityPosition(world, eid);
                    const loop = Sound.loop[eid] === 1;

                    const newTrack = soundManager.play(soundId, { volume, loop, x: pos.x, y: pos.y });
                    if (newTrack) {
                        activeAudios.set(eid, newTrack);
                        typeSet.add(eid);
                        newTrack.onEnded(() => handleStoppedSounds(world, eid));
                    }
                } else {
                    track.setVolume(volume * Sound.volume[eid]);
                }
            }
        }

        for (const eid of soundEids) {
            const state = Sound.state[eid];
            const track = activeAudios.get(eid);

            if (track) {
                if (state === SoundState.Stopped) {
                    soundManager.stopInstance(track);
                    activeAudios.delete(eid);
                    const type = Sound.type[eid] as SoundType;
                    soundsByType.get(type)?.delete(eid);
                    handleStoppedSounds(world, eid);
                } else if (state === SoundState.Paused && track.state === 'playing') {
                    soundManager.pauseInstance(track);
                } else if (state === SoundState.Playing && track.state === 'paused') {
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

function handleStoppedSounds(world: RenderGameWorld, eid: EntityId): void {
    const { DestroyOnSoundFinish, Parent, Children } = getRenderWorldComponents(world);
    if (!hasComponent(world, eid, DestroyOnSoundFinish)) return;

    // Sound entities are render-only (no Rapier atom): reap directly from RenderWorld,
    // disconnecting the parent->child edge if present.
    if (hasComponent(world, eid, Parent) && hasComponent(world, Parent.id[eid], Children)) {
        Children.removeChild(Parent.id[eid], eid);
    }
    removeEntity(world, eid);
}

export function disposeSoundSystem(): void {
    for (const [, track] of activeAudios) {
        soundManager.stopInstance(track);
    }
    activeAudios.clear();
}
