import { Vector2, World } from '@dimforge/rapier2d';

export function initPhysicalWorld() {
    let gravity = new Vector2(0, 0);
    return new World(gravity);
}

export type PhysicalWorld = World;