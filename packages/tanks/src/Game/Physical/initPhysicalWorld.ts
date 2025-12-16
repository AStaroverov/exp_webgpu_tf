import { Vector2, World } from '@dimforge/rapier2d-simd';
import { createRigidRectangle } from './createRigid';

export function initPhysicalWorld() {
    const gravity = new Vector2(0, 0);
    const world = new World(gravity);
    world.numSolverIterations = 4;

    // skip id == 0 because it's the default value for empty memory data
    createRigidRectangle(
        { width: 10, height: 10, x: 0, y: 0 },
        // @ts-ignore
        { physicalWorld: world }
    );

    return world;
}

export type PhysicalWorld = World;