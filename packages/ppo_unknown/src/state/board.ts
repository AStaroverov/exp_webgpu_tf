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

import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { NestedArray } from "renderer/src/utils.ts";
import { delegate } from "renderer/src/delegate.ts";
import { defineComponent } from "renderer/src/ECS/utils.ts";

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
 * addition to the network's positional encoding). Unit identity is a one-hot
 * `VehicleType` (`Type*` planes) + live reload state, written for self/ally/enemy cells.
 */
let C = 0;
export const BoardChannel = {
  /**
   * Normalized window COLUMN of the cell center: `col / (BOARD_COLS − 1)` → 0..1.
   * Written for EVERY in-view cell (pure window geometry, not gated by occupancy).
   */
  CoordX: C++,
  /**
   * Normalized window ROW of the cell center: `row / (BOARD_ROWS − 1)` → 0..1.
   * Written for EVERY in-view cell; pairs with `CoordX`.
   */
  CoordY: C++,
  /**
   * Enemy heat (0..1): per-cell max over ALL enemies of `1 − hexDist(cell, enemy)
   * / MAX_MAP_DIST`. The peak sits on the enemy's REAL current cell. This is how
   * enemies beyond the view radius are sensed: the in-window gradient points
   * toward them.
   */
  EnemyHeat: C++,
  /**
   * Not enterable / not visible (0/1): a static obstacle, an off-map cell, or a
   * cell beyond the view radius (the square window's corners).
   */
  Obstacle: C++,
  /** A reserved buffer cell around a unit or obstacle (grid `OccupantKind.Reserved`); 0/1. */
  Reserved: C++,
  /**
   * Under fire (0/1): a cell lying on the remaining flight path of an enemy
   * bullet currently in the air, OR on the predicted fire line of a visible
   * enemy whose queued action is `Fire`. Bullets travel a fixed distance in a
   * straight line, so the whole stretch a live bullet will still cross is marked
   * (not its current cell). See `markBulletThreat`.
   */
  UnderFire: C++,
  /** The observing agent's own cell — always the window center. */
  Self: C++,
  /** Same-team unit. */
  Ally: C++,
  /** Other-team unit. */
  Enemy: C++,
  /** Normalized hp (0..1) of the unit on the cell; 0 if no unit. */
  Hp: C++,
  /**
   * Damage statuses of the unit on the cell (flamethrower / freeze gun, §8 of
   * the stream-weapon design). `Dot` lives per-part in the game — aggregated
   * up to the vehicle here.
   */
  /** Damage-over-time: summed part dps over the max possible (alive parts x fire dps), 0..1. */
  Dot: C++,
  /**
   * The vehicle's freeze amount (`Slowed.slowMul`), 0..1: 0 = full speed
   * (the dense default), 1 = fully frozen — frost hits accumulate it.
   */
  Slow: C++,
  /**
   * Remaining reload time of the unit's gun, log-squashed:
   * `log1p(remaining_ms / 1000)` — 0 = ready to fire (or no reloading gun:
   * gunless / stream weapon), grows slowly with longer waits.
   */
  Reload: C++,
  /**
   * One-hot `VehicleType` of the unit on the cell (0/1 each, 0 if no unit) —
   * the type implies all static combat stats (mobility/damage/range/...), the
   * network learns them from the configs' behavior instead of hand-fed scalars.
   */
  TypeLightTank: C++,
  TypeMediumTank: C++,
  TypeRocketTank: C++,
  TypeFlameTank: C++,
  TypeFrostTank: C++,
  TypeEmpTank: C++,
} as const;

export const BOARD_CHANNELS = C;
export const BOARD_SIZE = BOARD_CELLS * BOARD_CHANNELS;

export type UnknownInputBoardComponent = ReturnType<typeof createUnknownInputBoardComponent>;

const instances = new WeakMap<World, UnknownInputBoardComponent>();

/** Per-world board store, lazily created on first touch (scenario setup runs first). */
export function ensureUnknownInputBoard(world: World): UnknownInputBoardComponent {
  let instance = instances.get(world);
  if (instance === undefined) {
    instance = createUnknownInputBoardComponent(world);
    instances.set(world, instance);
  }
  return instance;
}

export const createUnknownInputBoardComponent = defineComponent((UnknownInputBoard) => {
  const board = NestedArray.f64(BOARD_SIZE, delegate.defaultSize);
  function set(eid: number, row: number, col: number, channel: number, v: number) {
    const offset = (row * BOARD_COLS + col) * BOARD_CHANNELS + channel;
    board.set(eid, offset, v);
  }
  function get(eid: number, row: number, col: number, channel: number): number {
    const offset = (row * BOARD_COLS + col) * BOARD_CHANNELS + channel;
    return board.get(eid, offset);
  }

  return {
    board,

    addComponent(world: World, eid: number) {
      addComponent(world, eid, UnknownInputBoard);
      board.getBatch(eid).fill(0);
    },

    reset(eid: number) {
      board.getBatch(eid).fill(0);
    },

    set,
    get,

    /** Egocentric set: axial delta relative to the observer instead of row/col. */
    setDelta(eid: number, dq: number, dr: number, channel: number, v: number) {
      set(eid, dr + VIEW_RADIUS, dq + VIEW_RADIUS, channel, v);
    },

    getDelta(eid: number, dq: number, dr: number, channel: number): number {
      return get(eid, dr + VIEW_RADIUS, dq + VIEW_RADIUS, channel);
    },
  };
});
