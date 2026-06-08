/**
 * Hex grid configuration.
 *
 * Orientation: POINTY-top (vertex up). Rectangular bounds (cols x rows).
 * Coordinate system: honeycomb axial { q, r } (cube `s` is derived).
 *
 * The grid is a standalone structure built on `honeycomb-grid` (geometry,
 * neighbors, distance). It is intentionally decoupled from the game ECS world —
 * occupancy is tracked by reference (entity id + world id), see `HexGrid`.
 */

import { defineHex, Direction, Orientation } from 'honeycomb-grid';

export const HexGridConfig = {
    /** Hex "radius" in world pixels (xRadius === yRadius -> regular hex). */
    radius: 96,
    /** Grid size in hexes. */
    cols: 12,
    rows: 12,
    /** Orientation — pointy-top. */
    orientation: Orientation.POINTY,
} as const;

/**
 * The Hex class for this grid. honeycomb places hex [0,0] with its center at the
 * world origin (`origin: { x: 0, y: 0 }`), so `hex.x`/`hex.y` are world-space
 * centers directly (before any grid origin offset, see `HexGrid`).
 */
export const HexTile = defineHex({
    dimensions: HexGridConfig.radius,
    orientation: HexGridConfig.orientation,
    origin: { x: 0, y: 0 },
});

export type HexTile = InstanceType<typeof HexTile>;

/**
 * The 6 valid neighbor directions for a pointy-top hex: E/W + 4 diagonals.
 * (N and S are invalid for pointy orientation.)
 */
export const POINTY_DIRECTIONS: readonly Direction[] = [
    Direction.E,
    Direction.NE,
    Direction.NW,
    Direction.W,
    Direction.SW,
    Direction.SE,
];
