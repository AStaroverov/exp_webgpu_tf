/**
 * Sound System - manages audio playback for entities with Sound component
 * 
 * Strategy:
 * 1. Player entities always have priority for sounds
 * 2. Limits max simultaneous sounds per type
 * 3. Uses spatial audio with distance-based volume
 * 4. Cleans up sounds when entities are destroyed
 * 5. Handles parent-relative sounds (position follows parent)
 * 6. Auto-destroys sounds with DestroyOnSoundFinish when playback ends
 */

import { query, EntityId, hasComponent } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.js';
import { Sound, SoundType, SoundState, DestroyOnSoundFinish } from '../../Components/Sound.js';
import { CameraState } from '../Camera/CameraSystem.js';
import { soundManager } from './SoundManager.js';
import { WebAudioTrack } from './WebAudioTrack.js';
import { Parent } from '../../Components/Parent.js';
import { Destroy } from '../../Components/Destroy.js';
import { GlobalTransform, getMatrixTranslationX, getMatrixTranslationY } from 'renderer/src/ECS/Components/Transform.ts';
import { hypot } from '../../../../../../../lib/math.ts';

// Sound IDs for SoundManager
const SOUND_IDS: Record<SoundType, string> = {
    [SoundType.None]: '',
    [SoundType.TankMove]: 'tank_move',
    [SoundType.TankShoot]: 'tank_shoot',
    [SoundType.TankHit]: 'tank_hit',
    [SoundType.DebrisCollect]: 'debris_collect',
};

// Configuration
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

// Track active audio instances per entity
const activeAudios: Map<EntityId, WebAudioTrack> = new Map();

/**
 * Load all game sounds - call once at game start
 */
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
                // '/assets/sounds/tanks/hit/hit2.webm',
            ],
            maxInstances: CONFIG.maxSoundsPerType[SoundType.TankHit],
            volume: CONFIG.baseVolume[SoundType.TankHit],
            loop: false,
        }),
        // SoundManager.load(SOUND_IDS[SoundType.DebrisCollect], {
        //     src: '/assets/sounds/tanks/hit/an_explosive_shell_h_1.mp3',
        //     maxInstances: CONFIG.maxSoundsPerType[SoundType.DebrisCollect],
        //     volume: CONFIG.baseVolume[SoundType.DebrisCollect],
        //     loop: false,
        // }),
    ]);
}

/**
 * Get entity position from GlobalTransform
 * If entity has Parent, use parent's GlobalTransform
 * Otherwise use entity's own GlobalTransform
 */
function getEntityPosition(eid: EntityId, { world } = GameDI): { x: number; y: number } {
    // If entity follows a parent, use parent's GlobalTransform
    if (hasComponent(world, eid, Parent) && hasComponent(world, Parent.id[eid], GlobalTransform)) {
        const matrix = GlobalTransform.matrix.getBatch(Parent.id[eid]);
        return {
            x: getMatrixTranslationX(matrix),
            y: getMatrixTranslationY(matrix),
        };
    }
    
    // Use entity's own GlobalTransform
    if (hasComponent(world, eid, GlobalTransform)) {
        const matrix = GlobalTransform.matrix.getBatch(eid);
        return {
            x: getMatrixTranslationX(matrix),
            y: getMatrixTranslationY(matrix),
        };
    }
    
    return { x: 0, y: 0 };
}

/**
 * Calculate distance from camera to entity
 */
function getDistanceToCamera(eid: EntityId): number {
    const pos = getEntityPosition(eid);
    const dx = pos.x - CameraState.x;
    const dy = pos.y - CameraState.y;
    return hypot(dx, dy);
}

/**
 * Calculate volume based on distance (quadratic falloff)
 */
function calculateDistanceVolume(distance: number, baseVolume: number): number {
    if (distance <= CONFIG.refDistance) return baseVolume;
    if (distance >= CONFIG.hearingRange) return 0;

    const normalized = (distance - CONFIG.refDistance) / (CONFIG.hearingRange - CONFIG.refDistance);
    return baseVolume * (1 - normalized ** 3);
}

/**
 * Create the sound system
 */
export function createSoundSystem({ world } = GameDI) {
    // Track sounds per type for limiting
    const soundsByType: Map<SoundType, Set<EntityId>> = new Map();
    for (const type of Object.values(SoundType)) {
        if (typeof type === 'number') {
            soundsByType.set(type, new Set());
        }
    }

    return function updateSounds(_delta: number): void {
        // Update listener position
        soundManager.setListenerPosition(CameraState.x, CameraState.y);

        const soundEids = query(world, [Sound]);

        // Categorize entities by sound type and gather info
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

        // Gather all sound entities
        for (const eid of soundEids) {
            const type = Sound.type[eid] as SoundType;
            if (type === SoundType.None) continue;

            const distance = getDistanceToCamera(eid);
            const wantsToPlay = Sound.state[eid] === SoundState.Playing;

            entitiesByType.get(type)?.push({ eid, distance, wantsToPlay });
        }

        // Process each sound type
        for (const [type, entities] of entitiesByType) {
            const baseVolume = CONFIG.baseVolume[type];
            const maxSounds = CONFIG.maxSoundsPerType[type];
            const soundId = SOUND_IDS[type];
            const typeSet = soundsByType.get(type)!;

            // Filter to entities that want to play and are in range
            const playableEntities = entities
                .filter(e => e.wantsToPlay && (e.distance < CONFIG.hearingRange))
                .sort((a, b) => a.distance - b.distance);

            // Take only top N
            const topN = playableEntities.slice(0, maxSounds);
            const toPlay = new Set(topN.map(e => e.eid));

            // Stop sounds for entities that should no longer play
            for (const eid of typeSet) {
                if (toPlay.has(eid)) continue;
                const track = activeAudios.get(eid);
                track && soundManager.stopInstance(track);
                handleStoppedSounds(eid);
                activeAudios.delete(eid);
                typeSet.delete(eid);
            }

            // Start or update sounds for active entities
            for (const { eid, distance } of topN) {
                const volume = calculateDistanceVolume(distance, baseVolume);

                let track = activeAudios.get(eid);

                if (!track) {
                    // Start new sound - get position from entity
                    const pos = getEntityPosition(eid);
                    const loop = Sound.loop[eid] === 1;

                    const newTrack = soundManager.play(soundId, { volume, loop, x: pos.x, y: pos.y });
                    if (newTrack) {
                        activeAudios.set(eid, newTrack);
                        typeSet.add(eid);
                        newTrack.onEnded(() => handleStoppedSounds(eid));
                    }
                } else {
                    // Update volume
                    track.setVolume(volume * Sound.volume[eid]);
                }
            }
        }

        // Handle stopped/paused sounds
        for (const eid of soundEids) {
            const state = Sound.state[eid];
            const track = activeAudios.get(eid);

            if (track) {
                if (state === SoundState.Stopped) {
                    soundManager.stopInstance(track);
                    activeAudios.delete(eid);
                    const type = Sound.type[eid] as SoundType;
                    soundsByType.get(type)?.delete(eid);
                    handleStoppedSounds(eid);
                } else if (state === SoundState.Paused && track.state === 'playing') {
                    soundManager.pauseInstance(track);
                } else if (state === SoundState.Playing && track.state === 'paused') {
                    soundManager.resumeInstance(track);
                }
            }
        }

        // Cleanup for destroyed entities
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
