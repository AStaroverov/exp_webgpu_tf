export const ACTION_DIM = 4; // [shoot, move, rotate, turretRot]
export const TICK_TIME_SIMULATION = Math.round(16 * 1.5);
export const SNAPSHOT_EVERY = Math.round(200 / TICK_TIME_SIMULATION); // approx every 200ms = 6 * TICK_TIME_SIMULATION;

export const LEARNING_STEPS = 100_000_000;

if (SNAPSHOT_EVERY <= 2) {
    throw new Error('Setting SNAPSHOT_EVERY <= 2 is not allowed');
}
