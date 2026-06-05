/**
 * UnknownInputBoard — egocentric board observation storage (the `S` backing store),
 * mirroring tanks' `TankInputTensor` but holding a single multi-plane board instead
 * of physics groups + rays.
 *
 * Strategic representation only — NO physics (no world positions, velocities,
 * rotations, turret angles, rays, collider radii). Units live ON the hex grid, so
 * their position is encoded by their cell. Planes are POV-relative (self/ally/enemy
 * are relative to the observing agent's team).
 *
 * EGOCENTRIC window: the board is a (2R+1)×(2R+1) square of AXIAL deltas around the
 * observer — window col = dq + R, window row = dr + R, self always at the center.
 * Axial deltas (not row/col offsets) keep hex neighbor offsets parity-free and
 * hex distance a pure function of (dq, dr). Cells outside the map OR beyond the
 * view radius (hex distance > R; the square's corners) read as `Obstacle`.
 * Enemies beyond the view radius are still sensed through the `EnemyHeat` plane.
 *
 * Layout: one flat board buffer per observing entity, cell-major —
 *   offset = (row * COLS + col) * CHANNELS + channel
 * so it reshapes directly to `[ROWS, COLS, CHANNELS]` for a conv / cell-token input.
 * Stored values are raw game quantities (flags 0/1, hp already 0..1); any further
 * normalization happens later in the tensor-building step.
 */

import { addComponent } from 'bitecs';
import type { World } from 'bitecs';
import { NestedArray } from 'renderer/src/utils.ts';
import { delegate } from 'renderer/src/delegate.ts';
import { component } from 'renderer/src/ECS/utils.ts';

/** Visible distance in hex steps; the window spans [-R, R] in both axial axes. */
export const VIEW_RADIUS = 5;

export const BOARD_COLS = VIEW_RADIUS * 2 + 1; // 11 (axial dq + R)
export const BOARD_ROWS = VIEW_RADIUS * 2 + 1; // 11 (axial dr + R)
export const BOARD_CELLS = BOARD_COLS * BOARD_ROWS;

/** Hex distance of an axial delta from the window center. */
export function hexDeltaDistance(dq: number, dr: number): number {
    return (Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2;
}

/** Per-cell channels (board planes). Collapsed/trimmed per design decisions. */
export const BoardChannel = {
    /**
     * Not enterable / not visible (0/1): a static obstacle, an off-map cell, or a
     * cell beyond the view radius (the square window's corners).
     */
    Obstacle: 0,
    /** The observing agent's own cell — always the window center. */
    Self: 1,
    /** Same-team unit. */
    Ally: 2,
    /** Other-team unit. */
    Enemy: 3,
    /** Normalized hp (0..1) of the unit on the cell; 0 if no unit. */
    Hp: 4,
    /** A cell a unit is driving into (grid `OccupantKind.Reserved`); 0/1. */
    Reserved: 5,
    /**
     * Under fire (0/1): a cell lying on the remaining flight path of an enemy
     * bullet currently in the air. Bullets travel a fixed distance in a straight
     * line, so the whole stretch a live bullet will still cross is marked (not its
     * current cell). See `markBulletThreat`.
     */
    UnderFire: 6,
    /**
     * Enemy heat (0..1): per-cell max over ALL enemies (visible or not) of
     * `1 − hexDist(cell, enemy) / MAX_MAP_DIST`. 1 on the enemy's own cell,
     * fading to 0 at the farthest map distance; overlapping enemies keep the
     * max. This is how enemies beyond the view radius are sensed — the in-window
     * gradient points toward them.
     */
    EnemyHeat: 7,
} as const;

export const BOARD_CHANNELS = 8;
export const BOARD_SIZE = BOARD_CELLS * BOARD_CHANNELS;

export const UnknownInputBoard = component({
    board: NestedArray.f64(BOARD_SIZE, delegate.defaultSize),

    addComponent(world: World, eid: number) {
        addComponent(world, eid, UnknownInputBoard);
        UnknownInputBoard.board.getBatch(eid).fill(0);
    },

    reset(eid: number) {
        UnknownInputBoard.board.getBatch(eid).fill(0);
    },

    set(eid: number, row: number, col: number, channel: number, v: number) {
        const offset = (row * BOARD_COLS + col) * BOARD_CHANNELS + channel;
        UnknownInputBoard.board.set(eid, offset, v);
    },

    get(eid: number, row: number, col: number, channel: number): number {
        const offset = (row * BOARD_COLS + col) * BOARD_CHANNELS + channel;
        return UnknownInputBoard.board.get(eid, offset);
    },

    /** Egocentric set: axial delta relative to the observer instead of row/col. */
    setDelta(eid: number, dq: number, dr: number, channel: number, v: number) {
        UnknownInputBoard.set(eid, dr + VIEW_RADIUS, dq + VIEW_RADIUS, channel, v);
    },

    getDelta(eid: number, dq: number, dr: number, channel: number): number {
        return UnknownInputBoard.get(eid, dr + VIEW_RADIUS, dq + VIEW_RADIUS, channel);
    },
});
