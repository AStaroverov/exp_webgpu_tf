/**
 * Sound System - manages audio playback for entities with Sound component
 * 
 * Strategy:
 * 1. Player entities always have priority for sounds
 * 2. Limits max simultaneous sounds per type
 * 3. Uses spatial audio with distance-based volume
 * 4. Cleans up sounds when entities are destroyed
 */

import { query, EntityId, hasComponent } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { Sound, SoundType, SoundState } from '../../Components/Sound.ts';
import { RigidBodyState } from '../../Components/Physical.ts';
import { PlayerRef } from '../../Components/PlayerRef.ts';
import { CameraState } from '../Camera/CameraSystem.ts';
import { SoundManager } from './SoundManager.ts';

// Sound IDs for SoundManager
const SOUND_IDS: Record<SoundType, string> = {
    [SoundType.None]: '',
    [SoundType.TankMove]: 'tank_move',
    [SoundType.TankShoot]: 'tank_shoot',
    [SoundType.TankHit]: 'tank_hit',
};

// Configuration
const CONFIG = {
    maxSoundsPerType: {
        [SoundType.None]: 0,
        [SoundType.TankMove]: 5,
        [SoundType.TankShoot]: 8,
        [SoundType.TankHit]: 8,
    },
    hearingRange: 1000,   // Reduced from 1500 - sounds fade out faster
    refDistance: 100,     // Reduced from 200 - starts fading sooner
    nonPlayerVolumeMultiplier: 0.4,  // Non-player sounds are 40% of base volume
    baseVolume: {
        [SoundType.None]: 0,
        [SoundType.TankMove]: 0.3,
        [SoundType.TankShoot]: 0.35,  // Reduced by 30% (was 0.5)
        [SoundType.TankHit]: 0.28,    // Reduced by 30% (was 0.4)
    },
};

// Track active audio instances per entity
const activeAudios: Map<EntityId, HTMLAudioElement> = new Map();

/**
 * Load all game sounds - call once at game start
 */
export async function loadGameSounds(): Promise<void> {
    await Promise.all([
        SoundManager.load(SOUND_IDS[SoundType.TankMove], {
            src: '/assets/sounds/tanks/move/smartsound_TRANSPORTATION_TANK_Small_Tracks_Rattle_Fast_01.mp3',
            maxInstances: CONFIG.maxSoundsPerType[SoundType.TankMove] + 1,
            volume: CONFIG.baseVolume[SoundType.TankMove],
            loop: true,
        }),
        SoundManager.load(SOUND_IDS[SoundType.TankShoot], {
            src: [
                '/assets/sounds/tanks/shot/tank_shot_1.mp3',
                '/assets/sounds/tanks/shot/tank_shot_3.mp3',
                '/assets/sounds/tanks/shot/tank_shot_4.mp3',
            ],
            maxInstances: CONFIG.maxSoundsPerType[SoundType.TankShoot],
            volume: CONFIG.baseVolume[SoundType.TankShoot],
            loop: false,
        }),
        SoundManager.load(SOUND_IDS[SoundType.TankHit], {
            src: [
                '/assets/sounds/tanks/hit/an_explosive_shell_h_1.mp3',
                '/assets/sounds/tanks/hit/an_explosive_shell_h_2.mp3',
                '/assets/sounds/tanks/hit/an_explosive_shell_h_3.mp3',
                '/assets/sounds/tanks/hit/an_explosive_shell_h_4.mp3',
            ],
            maxInstances: CONFIG.maxSoundsPerType[SoundType.TankHit],
            volume: CONFIG.baseVolume[SoundType.TankHit],
            loop: false,
        }),
    ]);
}

/**
 * Calculate distance from camera to entity
 */
function getDistanceToCamera(eid: EntityId): number {
    const x = RigidBodyState.position.get(eid, 0);
    const y = RigidBodyState.position.get(eid, 1);
    const dx = x - CameraState.x;
    const dy = y - CameraState.y;
    return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculate volume based on distance (quadratic falloff)
 */
function calculateDistanceVolume(distance: number, baseVolume: number): number {
    if (distance <= CONFIG.refDistance) return baseVolume;
    if (distance >= CONFIG.hearingRange) return 0;

    const normalized = (distance - CONFIG.refDistance) / (CONFIG.hearingRange - CONFIG.refDistance);
    return baseVolume * (1 - normalized * normalized);
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
        SoundManager.setListenerPosition(CameraState.x, CameraState.y);

        const soundEids = query(world, [Sound]);

        // Categorize entities by sound type and gather info
        const entitiesByType: Map<SoundType, Array<{
            eid: EntityId;
            distance: number;
            isPlayer: boolean;
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
            const isPlayer = hasComponent(world, eid, PlayerRef);
            const wantsToPlay = Sound.state[eid] === SoundState.Playing;

            entitiesByType.get(type)?.push({ eid, distance, isPlayer, wantsToPlay });
        }

        // Process each sound type
        for (const [type, entities] of entitiesByType) {
            const maxSounds = CONFIG.maxSoundsPerType[type];
            const soundId = SOUND_IDS[type];
            const baseVolume = CONFIG.baseVolume[type];
            const typeSet = soundsByType.get(type)!;

            // Filter to entities that want to play and are in range
            const playableEntities = entities.filter(e =>
                e.wantsToPlay && (e.distance < CONFIG.hearingRange || e.isPlayer),
            );

            // Sort: players first, then by distance
            playableEntities.sort((a, b) => {
                if (a.isPlayer && !b.isPlayer) return -1;
                if (!a.isPlayer && b.isPlayer) return 1;
                return a.distance - b.distance;
            });

            // Take only top N
            const toPlay = new Set(playableEntities.slice(0, maxSounds).map(e => e.eid));

            // Stop sounds for entities that should no longer play
            for (const eid of typeSet) {
                if (!toPlay.has(eid)) {
                    const audio = activeAudios.get(eid);
                    if (audio) {
                        SoundManager.stopInstance(audio);
                        activeAudios.delete(eid);
                    }
                    typeSet.delete(eid);
                }
            }

            // Start or update sounds for active entities
            for (const { eid, distance, isPlayer } of playableEntities.slice(0, maxSounds)) {
                const volume = isPlayer
                    ? baseVolume
                    : calculateDistanceVolume(distance, baseVolume) * CONFIG.nonPlayerVolumeMultiplier;

                let audio = activeAudios.get(eid);

                if (!audio) {
                    // Start new sound
                    const x = RigidBodyState.position.get(eid, 0);
                    const y = RigidBodyState.position.get(eid, 1);
                    const loop = Sound.loop[eid] === 1;

                    const newAudio = SoundManager.play(soundId, { volume, loop, x, y });
                    if (newAudio) {
                        activeAudios.set(eid, newAudio);
                        typeSet.add(eid);
                    }
                } else {
                    // Update volume
                    audio.volume = volume * Sound.volume[eid];
                }
            }
        }

        // Handle stopped/paused sounds
        for (const eid of soundEids) {
            const state = Sound.state[eid];
            const audio = activeAudios.get(eid);

            if (audio) {
                if (state === SoundState.Stopped) {
                    SoundManager.stopInstance(audio);
                    activeAudios.delete(eid);
                    const type = Sound.type[eid] as SoundType;
                    soundsByType.get(type)?.delete(eid);
                } else if (state === SoundState.Paused && !audio.paused) {
                    audio.pause();
                } else if (state === SoundState.Playing && audio.paused) {
                    audio.play().catch(() => {});
                }
            }
        }

        // Cleanup for destroyed entities
        for (const [eid, audio] of activeAudios) {
            if (!soundEids.includes(eid)) {
                SoundManager.stopInstance(audio);
                activeAudios.delete(eid);
                for (const typeSet of soundsByType.values()) {
                    typeSet.delete(eid);
                }
            }
        }
    };
}

/**
 * Dispose sound system
 */
export function disposeSoundSystem(): void {
    for (const [, audio] of activeAudios) {
        SoundManager.stopInstance(audio);
    }
    activeAudios.clear();
}
