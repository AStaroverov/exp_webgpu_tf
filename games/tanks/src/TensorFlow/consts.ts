export const TANK_RADIUS = 80;
export const TANK_COUNT_SIMULATION = 6; // Reduced to make training more manageable
export const TICK_TIME_REAL = 1;
export const TICK_TIME_SIMULATION = 16.6667 * 2;
export const MAX_STEPS = 3000; // Limit episode length
export const INPUT_DIM = 63; // Tank state dimensions (same as your original implementation)
export const ACTION_DIM = 5; // [shoot, move, turn, targetX, targetY]