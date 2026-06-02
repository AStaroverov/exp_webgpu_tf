/**
 * ActionSlot — slot schema for the `ActionsQueue` component (actions live on the
 * controlled entity as a fixed-size queue). Holds the queue bounds, the per-kind
 * param offsets, and the fire phases. The cross-buffer queue shift now lives on the
 * component itself (`ActionsQueue.dropFront`), so the buffers stay encapsulated.
 */

/** Bounded per-owner queue depth: one running action + one pre-decided next. */
export const MAX_QUEUE = 2;

/** Per-kind param/scratch floats per slot (≤2 used today, headroom). */
export const PARAMS = 4;

/** Per-kind `params` offsets (`p0..p3`). */
export const MoveStepParamOffset = {
    /** Steering speed toward the target hex. */
    speed: 0,
} as const;

export const AimParamOffset = {
    /** Aim accuracy (radians). */
    tolerance: 0,
} as const;

export const FireParamOffset = {
    /** Fire phase: 0 = aiming, 1 = waitReady, 2 = firing. */
    phase: 0,
} as const;

export const HoldParamOffset = {
    /** Hold duration (ms). */
    duration: 0,
    /** Accumulated elapsed time (ms). */
    elapsed: 1,
} as const;

/** Fire phases (stored in the `phase` param): aim the turret, wait for reload, fire. */
export const FIRE_PHASE_AIMING = 0;
export const FIRE_PHASE_WAIT_READY = 1;
export const FIRE_PHASE_FIRING = 2;
