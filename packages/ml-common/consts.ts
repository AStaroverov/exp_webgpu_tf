export const ACTION_DIM = 4; // [shoot, move, rotate, turretRot]
export const LOG_STD_DIM = ACTION_DIM; // [shoot, move, rotate, turretRot]
export const TICK_TIME_SIMULATION = Math.round(16 * 1.5);
export const SNAPSHOT_EVERY = Math.round(100 / TICK_TIME_SIMULATION); // approx every 100ms = 3 * TICK_TIME_SIMULATION;

export const LEARNING_STEPS = 200_000_000;

if (SNAPSHOT_EVERY <= 2) {
    throw new Error('Setting SNAPSHOT_EVERY <= 2 is not allowed');
}
