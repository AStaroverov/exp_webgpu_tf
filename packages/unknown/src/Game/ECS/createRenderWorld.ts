import { createWorld, World } from 'bitecs';
import { createRenderComponents, RenderComponents } from '../../../../renderer/src/ECS/world.ts';
import { Opaque } from '../../../../renderer/src/type.ts';

import { createChildrenComponent } from './Components/Children.ts';
import { createParentComponent } from './Components/Parent.ts';
import { createExhaustPipeComponent } from './Components/ExhaustPipe.ts';
import { createPhysicsRefComponent } from '../DI/links.ts';

function createRenderOnlyComponents(world: World) {
    return {
        Parent: createParentComponent(world),
        Children: createChildrenComponent(world),
        ExhaustPipe: createExhaustPipeComponent(world),
        PhysicsRef: createPhysicsRefComponent(world),
    };
}

export type RenderWorldComponents = RenderComponents & ReturnType<typeof createRenderOnlyComponents>;

export type RenderGameWorld = Opaque<'RenderGameWorld', World<{
    components: RenderWorldComponents;
    time: {
        delta: number;
        elapsed: number;
        then: number;
    };
}>>;

export function createRenderWorld(): RenderGameWorld {
    const context = {
        components: null as unknown as RenderWorldComponents,
        time: {
            delta: 0,
            elapsed: 0,
            then: performance.now(),
        },
    };
    const world = createWorld(context) as unknown as RenderGameWorld;
    context.components = {
        ...createRenderComponents(world),
        ...createRenderOnlyComponents(world),
    };
    return world;
}

export function getRenderWorldComponents(world: RenderGameWorld): RenderWorldComponents {
    const components = world.components;
    if (!components) {
        throw new Error('Render components are not available on this world');
    }
    return components;
}
