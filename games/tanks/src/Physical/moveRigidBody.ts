import { vec2 } from 'gl-matrix';
import { RigidBody } from '@dimforge/rapier2d-simd/rapier';

const direction = vec2.create();
const translation = { x: 0, y: 0 };

export function moveRigidBody(body: RigidBody, speed: number): void {
    const angle = body.rotation(); // Get current rotation in radians
    direction[0] = Math.sin(angle);
    direction[1] = Math.cos(angle);

    // Scale direction by speed
    vec2.scale(direction, direction, speed);

    // Get current position
    const position = body.translation();

    // Apply new position
    translation.x = position.x + direction[0];
    translation.y = position.y + direction[1];
    body.setTranslation(translation, true);
}

export function moveRigidBodyByVector(body: RigidBody, x: number, y: number): void {
    const direction = vec2.fromValues(x, y);
    const speed = vec2.length(direction);
    // Normalize the direction to ensure consistent movement speed
    const normalizedDirection = vec2.create();
    vec2.normalize(normalizedDirection, direction);

    // Scale by speed
    vec2.scale(normalizedDirection, normalizedDirection, speed);

    // Get current position
    const position = body.translation();
    const newPosition = vec2.fromValues(position.x + normalizedDirection[0], position.y + normalizedDirection[1]);

    // Apply new position
    body.setTranslation({ x: newPosition[0], y: newPosition[1] }, true);
}

export function moveRigidBodyForward(body: RigidBody): void {
    const angle = body.rotation(); // Get current rotation in radians

    // Compute movement direction from angle
    const direction = vec2.fromValues(Math.cos(angle), Math.sin(angle));

    // Scale movement by delta time
    vec2.scale(direction, direction, 1);

    // Get current position
    const position = body.translation();
    const newPosition = vec2.fromValues(position.x + direction[0], position.y + direction[1]);

    // Apply new position
    body.setTranslation({ x: newPosition[0], y: newPosition[1] }, true);
}

export function moveRigidBodyForward1(body: RigidBody, speed: number = -1): void {
    const angle = body.rotation(); // Get current rotation in radians

    // Compute movement direction where 0° (0 radians) moves up (Y+)
    const direction = vec2.fromValues(-Math.sin(angle), Math.cos(angle));

    // Scale movement by delta time
    vec2.scale(direction, direction, speed);

    // Get current position
    const position = body.translation();
    const newPosition = vec2.fromValues(position.x + direction[0], position.y + direction[1]);

    // Apply new position
    body.setTranslation({ x: newPosition[0], y: newPosition[1] }, true);
}
