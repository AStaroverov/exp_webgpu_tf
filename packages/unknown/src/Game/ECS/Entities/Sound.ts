import { addEntity, EntityId } from 'bitecs';
import { SoundType } from '../Components/Sound.ts';
import { getSoundWorldComponents } from '../createSoundWorld.ts';
import { addTransformComponents, applyMatrixTranslate } from 'renderer/src/ECS/Components/Transform.ts';
import { setSoundOwner } from '../refs.ts';
import { Worlds } from '../../DI/Worlds.ts';

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

export interface SpawnSoundForOwnerOptions extends SpawnSoundOptions {
    ownerEid: EntityId;
}

// Fixed-position sound: bakes the world position into SoundWorld's own transform.
export function spawnSoundAtPosition(
    options: SpawnSoundAtPositionOptions,
    { soundWorld } = Worlds,
): EntityId {
    const { Sound, DestroyOnSoundFinish, LocalTransform } = getSoundWorldComponents(soundWorld);
    const eid = addEntity(soundWorld);

    addTransformComponents(soundWorld, eid);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), options.x, options.y, 0);

    Sound.addComponent(soundWorld, eid, options.type, {
        loop: options.loop ?? false,
        volume: options.volume ?? 1,
        autoplay: options.autoplay ?? true,
    });

    if (options.destroyOnFinish) {
        DestroyOnSoundFinish.addComponent(soundWorld, eid);
    }

    return eid;
}

// Owner-linked sound: follows the owning physics atom (SoundOwnerRef replaces the old
// render Parent/Children edge). Position resolves via the owner atom's RigidBodyState.
export function spawnSoundForOwner(
    options: SpawnSoundForOwnerOptions,
    { soundWorld } = Worlds,
): EntityId {
    const { Sound, DestroyOnSoundFinish } = getSoundWorldComponents(soundWorld);
    const eid = addEntity(soundWorld);

    setSoundOwner(eid, options.ownerEid);

    Sound.addComponent(soundWorld, eid, options.type, {
        loop: options.loop ?? true,
        volume: options.volume ?? 1,
        autoplay: options.autoplay ?? true,
    });

    if (options.destroyOnFinish) {
        DestroyOnSoundFinish.addComponent(soundWorld, eid);
    }

    return eid;
}
