import { Vector2, World } from '@dimforge/rapier2d-simd';

export function initPhysicalWorld() {
    const gravity = new Vector2(0, 0);
    const world = new World(gravity);
    world.numSolverIterations = 4;

    return world;
}

export type PhysicalWorld = World;