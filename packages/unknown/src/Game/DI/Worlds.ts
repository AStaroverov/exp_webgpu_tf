import { PhysicsWorld } from '../ECS/createPhysicsWorld.ts';
import { RenderGameWorld } from '../ECS/createRenderWorld.ts';
import { ActionWorld } from '../ECS/Actions/createActionWorld.ts';
import { PhysicalWorld } from '../Physical/initPhysicalWorld.ts';

/**
 * Worlds — the single registry of every ECS/physics world in the game, each with
 * its own (Opaque-branded) type. Set in `createGame`, nulled on teardown.
 *
 * Functions that need a world take it as a trailing optional destructured default,
 * e.g. `fn(eid, ..., { physicsWorld } = Worlds)`, so call sites stay terse while the
 * branded types keep the wrong world from being passed in.
 */
export const Worlds: {
    physicsWorld: PhysicsWorld;
    renderWorld: RenderGameWorld;
    physicalWorld: PhysicalWorld;
    actionWorld: ActionWorld;
} = {
    physicsWorld: null!,
    renderWorld: null!,
    physicalWorld: null!,
    actionWorld: null!,
};
