import { createWorld, World } from 'bitecs';
import { Opaque } from '../../../../renderer/src/type.ts';
import { createRenderComponents } from '../../../../renderer/src/ECS/world.ts';
import {
    createLocalTransformComponent,
    createGlobalTransformComponent,
} from '../../../../renderer/src/ECS/Components/Transform.ts';

import { createVFXComponent } from './Components/VFX.ts';
import { createTreadMarkComponent } from './Components/TreadMark.ts';
import { createProgressComponent } from './Components/Progress.ts';
import { createDestroyByTimeoutComponent } from './Components/Destroy.ts';

// The fx world: short-lived visual effects (tread marks, explosions, muzzle/hit
// flashes, exhaust smoke), carved out of RenderWorld. fx are spawned from already
// resolved WORLD-SPACE snapshots (no Parent/Children), so each fx entity carries a
// flat Local->Global transform.
//
// CRITICAL (transform-singleton aliasing): renderer's `LocalTransform`/`GlobalTransform`
// are module-level singletons keyed by raw eid and shared across worlds. FxWorld MUST
// get its OWN transform instances so FxWorld eids don't alias RenderWorld eids in the
// shared transform store. We therefore override the singletons that `createRenderComponents`
// returns with fresh per-world instances (Color/Shape/Roundness are already per-world).
function createFxOnlyComponents(world: World) {
    return {
        ...createRenderComponents(world),
        // per-world transforms (NOT the shared singletons)
        LocalTransform: createLocalTransformComponent(),
        GlobalTransform: createGlobalTransformComponent(),
        // fx components
        VFX: createVFXComponent(world),
        TreadMark: createTreadMarkComponent(world),
        ProgressFx: createProgressComponent(world),
        DestroyByTimeoutFx: createDestroyByTimeoutComponent(world),
    };
}

export type FxWorldComponents = ReturnType<typeof createFxOnlyComponents>;

export type FxWorld = Opaque<'FxWorld', World<{
    components: FxWorldComponents;
    time: {
        delta: number;
        elapsed: number;
        then: number;
    };
}>>;

export function createFxWorld(): FxWorld {
    const context = {
        components: null as unknown as FxWorldComponents,
        time: {
            delta: 0,
            elapsed: 0,
            then: performance.now(),
        },
    };
    const world = createWorld(context) as unknown as FxWorld;
    context.components = createFxOnlyComponents(world);
    return world;
}

export function getFxWorldComponents(world: FxWorld): FxWorldComponents {
    const components = world.components;
    if (!components) {
        throw new Error('Fx components are not available on this world');
    }
    return components;
}
