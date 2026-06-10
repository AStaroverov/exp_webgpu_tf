import { createHeadlightSet, createRectangleSet, PartsData } from '../Common/TankParts.ts';

export const SIZE = 6;
export const PADDING = SIZE + 1;

// Elongated hull: clearly longer (14 along +X / forward) than wide (6) so the
// vehicle reads as a launcher chassis rather than a tank.
export const HULL_COLS = 14;
export const HULL_ROWS = 6;
export const hullSet = createRectangleSet(HULL_COLS, HULL_ROWS, SIZE, PADDING);
export const headlightSet = createHeadlightSet(HULL_COLS, HULL_ROWS, SIZE, PADDING);

/** Shift every part in a set by (dx, dy) in the parent's local frame. */
function offsetSet(set: PartsData[], dx: number, dy: number): PartsData[] {
    return set.map(([x, y, w, h]) => [x + dx, y + dy, w, h] as PartsData);
}

// Lateral offset of the two assemblies from the hull centerline (~1.5 cells to
// each side), so the rail and cabin sit on opposite sides of center.
const SIDE_OFFSET = PADDING * 1.5;

// Launch rail (replaces the turret): 2 wide, running the full tank length, on
// the LEFT side and shifted one cell rearward so it overhangs the back.
export const RAIL_COLS = HULL_COLS;
export const RAIL_ROWS = 2;
export const railSet = offsetSet(
    createRectangleSet(RAIL_COLS, RAIL_ROWS, SIZE, PADDING),
    -PADDING,    // one cell back → overhangs the rear (-X)
    SIDE_OFFSET, // left of center
);

// Pilot cabin (replaces the gun): a compact 2x2 block on the RIGHT of center.
export const CABIN_COLS = 3;
export const CABIN_ROWS = 3;
export const cabinSet = offsetSet(
    createRectangleSet(CABIN_COLS, CABIN_ROWS, SIZE, PADDING),
    PADDING * 3,
    -SIDE_OFFSET, // right of center
);

// World-space front of the rail (its forward edge), used as the rocket muzzle.
export const RAIL_FRONT_X = (HULL_COLS / 2 - 1) * PADDING;
export const RAIL_Y = SIDE_OFFSET;

export const CATERPILLAR_SIZE = 3;
export const CATERPILLAR_PADDING = CATERPILLAR_SIZE + 1;
export const CATERPILLAR_LINE_COUNT = 26;
export const caterpillarLength = CATERPILLAR_LINE_COUNT * CATERPILLAR_PADDING;

// Track anchor (distance from hull center to track center): just outside the
// hull's half-height (HULL_ROWS / 2 * PADDING) so the tracks frame the chassis.
export const TRACK_ANCHOR_Y = PADDING * (HULL_ROWS / 2) + PADDING;

export const caterpillarSet = createRectangleSet(
    CATERPILLAR_LINE_COUNT, 2,
    CATERPILLAR_SIZE, CATERPILLAR_PADDING,
    SIZE, PADDING,
);

export const caterpillarSetLeft: PartsData[] = caterpillarSet;
export const caterpillarSetRight: PartsData[] = caterpillarSet;

export const PARTS_COUNT =
    hullSet.length + headlightSet.length + railSet.length + cabinSet.length
    + CATERPILLAR_LINE_COUNT * 2 * 2;
