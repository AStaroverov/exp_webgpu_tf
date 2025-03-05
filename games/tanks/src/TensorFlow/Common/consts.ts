export const TANK_COUNT_SIMULATION = 6;
export const TANK_RADIUS = 80;

export const TICK_TIME_REAL = 1;
export const TICK_TIME_SIMULATION = 16.6667 * 2;
export const SKIP_TICKS = 2;
export const MAX_STEPS = 2000; // Limit episode length
export const INPUT_DIM = 63; // Tank state dimensions (same as your original implementation)
export const ACTION_DIM = 5; // [shoot, moveX, moveY, targetX, targetY]