import { createWorld, World } from 'bitecs';
import { Opaque } from '../../../../renderer/src/type.ts';
import {
    createLocalTransformComponent,
    createGlobalTransformComponent,
} from '../../../../renderer/src/ECS/Components/Transform.ts';

import {
    createSoundComponent,
    createDestroyOnSoundFinishComponent,
    createSoundParentRelativeComponent,
} from './Components/Sound.ts';
import { createSoundOwnerRefComponent } from '../DI/links.ts';

// The sound world: positional audio emitters (engine-move, shoot, hit), carved out of
// RenderWorld. Owner-linked sounds resolve their world position via the owning physics
// atom (SoundOwnerRef -> atom RigidBodyState); fixed-position sounds bake their world
// position into the per-world transform below.
//
// CRITICAL (transform-singleton aliasing): renderer's `LocalTransform`/`GlobalTransform`
// are module-level singletons keyed by raw eid and shared across worlds. SoundWorld MUST
// get its OWN transform instances so SoundWorld eids don't alias RenderWorld/FxWorld eids
// in the shared transform store.
function createSoundOnlyComponents(world: World) {
    return {
        // per-world transforms for fixed-position sounds (NOT the shared singletons)
        LocalTransform: createLocalTransformComponent(),
        GlobalTransform: createGlobalTransformComponent(),
        // sound components
        Sound: createSoundComponent(world),
        SoundParentRelative: createSoundParentRelativeComponent(world),
        DestroyOnSoundFinish: createDestroyOnSoundFinishComponent(world),
        // sound -> owner atom link (written only via setSoundOwner)
        SoundOwnerRef: createSoundOwnerRefComponent(world),
    };
}

export type SoundWorldComponents = ReturnType<typeof createSoundOnlyComponents>;

export type SoundWorld = Opaque<'SoundWorld', World<{
    components: SoundWorldComponents;
    time: {
        delta: number;
        elapsed: number;
        then: number;
    };
}>>;

export function createSoundWorld(): SoundWorld {
    const context = {
        components: null as unknown as SoundWorldComponents,
        time: {
            delta: 0,
            elapsed: 0,
            then: performance.now(),
        },
    };
    const world = createWorld(context) as unknown as SoundWorld;
    context.components = createSoundOnlyComponents(world);
    return world;
}

export function getSoundWorldComponents(world: SoundWorld): SoundWorldComponents {
    const components = world.components;
    if (!components) {
        throw new Error('Sound components are not available on this world');
    }
    return components;
}
