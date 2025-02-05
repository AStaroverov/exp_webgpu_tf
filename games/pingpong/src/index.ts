import { Vector2, World } from '@dimforge/rapier2d';

export function initPhysicalWorld() {
    let gravity = new Vector2(0.0, 9.81);
    let world = new World(gravity);

    return world;
}

export type PhysicalWorld = World;