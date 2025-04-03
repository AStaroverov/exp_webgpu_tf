import { TANK_APPROXIMATE_COLLISION_RADIUS } from '../../ECS/Components/Tank.ts';

export const TANK_COUNT_SIMULATION_MIN = 2;
export const TANK_COUNT_SIMULATION_MAX = 6;
export const TANK_RADIUS = TANK_APPROXIMATE_COLLISION_RADIUS;

export const TICK_TIME_REAL = 1;
export const TICK_TIME_SIMULATION = 16.6667 * 2;
export const TICK_TRAIN_TIME = 10;
export const INPUT_DIM = 85; // Tank state dimensions (same as your original implementation)
export const ACTION_DIM = 5; // [shoot, moveX, moveY, targetX, targetY]