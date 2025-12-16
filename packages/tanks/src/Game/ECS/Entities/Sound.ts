/**
 * SoundEntity - ECS-based sound spawning
 * 
 * Creates sound entities with proper component composition:
 * - Sound: core sound data (type, volume, loop, state)
 * - Transform: for positional sounds (position is read from LocalTransform)
 * - Parent: for sounds that follow an entity's position
 * - SoundParentRelative: sound lifecycle is tied to parent (destroyed when parent is destroyed)
 * - DestroyOnSoundFinish: for one-shot sounds that should be cleaned up after playback
 */

import { addEntity, EntityId, hasComponent } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { Sound, SoundType, DestroyOnSoundFinish } from '../Components/Sound.ts';
import { Parent } from '../Components/Parent.ts';
import { Children } from '../Components/Children.ts';
import { addTransformComponents, applyMatrixTranslate, LocalTransform } from 'renderer/src/ECS/Components/Transform.ts';

export interface SpawnSoundOptions {
    type: SoundType;
    volume?: number;
    loop?: boolean;
    autoplay?: boolean;
    destroyOnFinish?: boolean;
}

export interface SpawnSoundAtPositionOptions extends SpawnSoundOptions {
    x: number;
    y: number;
}

export interface SpawnSoundWithParentOptions extends SpawnSoundOptions {
    parentEid: EntityId;
}

/**
 * Spawn a sound at a specific position (one-shot positional sound)
 * The sound entity will be destroyed when playback finishes.
 */
export function spawnSoundAtPosition(
    options: SpawnSoundAtPositionOptions,
    { world } = GameDI,
): EntityId {
    const eid = addEntity(world);

    // Add transform for position
    addTransformComponents(world, eid);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, 0);

    // Add sound component
    Sound.addComponent(world, eid, options.type, {
        loop: options.loop ?? false,
        volume: options.volume ?? 1,
        autoplay: options.autoplay ?? true,
    });

    if (options.destroyOnFinish) {
        DestroyOnSoundFinish.addComponent(world, eid);
    }

    return eid;
}

/**
 * Spawn a sound attached to a parent entity (follows parent position)
 * Use for continuous sounds like engine, movement, etc.
 */
export function spawnSoundAtParent(
    options: SpawnSoundWithParentOptions,
    { world } = GameDI,
): EntityId {
    const eid = addEntity(world);

    // Ensure parent has Children component
    if (!hasComponent(world, options.parentEid, Children)) {
        Children.addComponent(world, options.parentEid);
    }

    // Add parent reference
    Parent.addComponent(world, eid, options.parentEid);
    Children.addChildren(options.parentEid, eid);

    // Add sound component
    Sound.addComponent(world, eid, options.type, {
        loop: options.loop ?? true,
        volume: options.volume ?? 1,
        autoplay: options.autoplay ?? true,
    });

    if (options.destroyOnFinish) {
        DestroyOnSoundFinish.addComponent(world, eid);
    }

    return eid;
}
