/**
 * UnknownInputBoard — chess-like board observation storage (the `S` backing store),
 * mirroring tanks' `TankInputTensor` but holding a single multi-plane board instead
 * of physics groups + rays.
 *
 * Strategic representation only — NO physics (no world positions, velocities,
 * rotations, turret angles, rays, collider radii). Units live ON the 8x8 grid, so
 * their position is encoded by their cell. Planes are POV-relative (self/ally/enemy
 * are relative to the observing agent's team).
 *
 * Layout: one flat board buffer per observing entity, cell-major —
 *   offset = (row * COLS + col) * CHANNELS + channel
 * so it reshapes directly to `[ROWS, COLS, CHANNELS]` for a conv / cell-token input.
 * Stored values are raw game quantities (flags 0/1, hp already 0..1); any further
 * normalization happens later in the tensor-building step.
 */

import { addComponent, World } from 'bitecs';
import { NestedArray } from 'renderer/src/utils.ts';
import { delegate } from 'renderer/src/delegate.ts';
import { component } from 'renderer/src/ECS/utils.ts';
import { HexGridConfig } from '../../../unknown/src/Game/Map/HexConfig.ts';

export const BOARD_COLS = HexGridConfig.cols; // 8
export const BOARD_ROWS = HexGridConfig.rows; // 8
export const BOARD_CELLS = BOARD_COLS * BOARD_ROWS;

/** Per-cell channels (board planes). Collapsed/trimmed per design decisions. */
export const BoardChannel = {
    /** Static obstacle on the cell (collapsed passable/obstacle: obstacle=1 → impassable). */
    Obstacle: 0,
    /** The observing agent's own cell. */
    Self: 1,
    /** Same-team unit. */
    Ally: 2,
    /** Other-team unit. */
    Enemy: 3,
    /** Normalized hp (0..1) of the unit on the cell; 0 if no unit. */
    Hp: 4,
} as const;

export const BOARD_CHANNELS = 5;
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
});
