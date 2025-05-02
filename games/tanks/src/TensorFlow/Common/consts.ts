import { TANK_APPROXIMATE_COLLISION_RADIUS } from '../../ECS/Components/Tank.ts';

export const ACTION_DIM = 5; // [shoot, moveX, moveY, targetX, targetY]
export const TANK_RADIUS = TANK_APPROXIMATE_COLLISION_RADIUS;
export const TICK_TIME_SIMULATION = Math.round(16 * 1.5);
export const SNAPSHOT_EVERY = Math.round(100 / TICK_TIME_SIMULATION); // approx every 100ms = 3 * TICK_TIME_SIMULATION;

if (SNAPSHOT_EVERY <= 2) {
    throw new Error('Setting SNAPSHOT_EVERY <= 2 is not allowed');
}
