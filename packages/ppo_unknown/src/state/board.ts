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

/**
 * Per-cell channels (board planes). Everything lives in this ONE map: spatial /
 * strategic planes AND per-unit identity (class + combat stats) written on the
 * unit's own cell — a unit's position IS its cell, so no separate token store is
 * needed. The `CoordX`/`CoordY` planes carry normalized window geometry (kept in
 * addition to the network's positional encoding). Stat channels
 * (`Role..Range`) come from `vehicleStats.ts`, written for self/ally/enemy cells.
 */
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
     * bullet currently in the air, OR on the predicted fire line of a visible
     * enemy whose queued action is `Fire`. Bullets travel a fixed distance in a
     * straight line, so the whole stretch a live bullet will still cross is marked
     * (not its current cell). See `markBulletThreat`.
     */
    UnderFire: 6,
    /**
     * Enemy heat (0..1): per-cell max over enemies KNOWN to the observer's team
     * (spotting confidence > 0) of `confidence · (1 − hexDist(cell, enemy) /
     * MAX_MAP_DIST)`. The peak sits on the enemy's REAL current cell (positions
     * stay honest — no stale ghosts) and is scaled by his fading spotting weight:
     * 1 right after a spot, decaying to 0 over a 3 s memory window. Unspotted
     * enemies (confidence 0) contribute nothing. This is how enemies beyond the
     * view radius — and recently-lost ones — are sensed: the in-window gradient
     * points toward them while it lasts.
     */
    EnemyHeat: 7,
    /**
     * Normalized window COLUMN of the cell center: `col / (BOARD_COLS − 1)` → 0..1.
     * Written for EVERY in-view cell (pure window geometry, not gated by occupancy).
     */
    CoordX: 8,
    /**
     * Normalized window ROW of the cell center: `row / (BOARD_ROWS − 1)` → 0..1.
     * Written for EVERY in-view cell; pairs with `CoordX`.
     */
    CoordY: 9,
    /**
     * Fading spotting confidence (0..1), a single per-victim value (two-sided game).
     * On an ENEMY cell: `Spottable.getConfidence` (the spot source + freshness — beam 1,
     * fire 0.5, proximity 0.25, fading from there), gated by `isVisible`. On the SELF
     * cell the same call reads "am I spotted by the enemy". 0 elsewhere; the
     * `Self`/`Enemy` planes disambiguate which meaning applies.
     */
    SpotConfidence: 10,
    /**
     * Under beam (0/1): a hex geometrically intersecting any Ranger searchlight
     * beam this tick (the swept capsule from `Beam.getBeamTargets`).
     * Always written, regardless of whether the emitting Ranger is itself spotted —
     * light is physically visible, same principle as live bullets in `UnderFire`.
     */
    UnderBeam: 11,
    /**
     * Per-unit identity + combat stats of the unit on the cell (0 if no unit), from
     * `vehicleStats.ts` — written for self/ally/visible-enemy cells, same normalizers
     * the units vector used. `Role`: 0 = fighter (gun), 1 = scout (Ranger).
     */
    Role: 12,
    /** Engine speed 0..1. */
    Mobility: 13,
    /** Gun damage 0..1 (0 if gunless). */
    Firepower: 14,
    /** Reload speed 0..1, faster = higher (0 if gunless). */
    Reload: 15,
    /** Bullet range 0..1 (0 if gunless). */
    Range: 16,
} as const;

export const BOARD_CHANNELS = 17;
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
