import { Vector2, World } from '@dimforge/rapier2d-simd';

export function initPhysicalWorld() {
    let gravity = new Vector2(0, 0);
    return new World(gravity);
}

export type PhysicalWorld = World;