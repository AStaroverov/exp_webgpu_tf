import { createRectangleSet, PartsData } from '../Common/TankParts.ts';

export const DENSITY = 275;
export const SIZE = 5;
export const PADDING = SIZE + 1;

export const hullSet = createRectangleSet(10, 8, SIZE, PADDING);
export const turretHeadSet = createRectangleSet(6, 5, SIZE, PADDING);
export const turretGunSet = createRectangleSet(6, 2, SIZE, PADDING);

export const CATERPILLAR_SIZE = 3;
export const CATERPILLAR_PADDING = CATERPILLAR_SIZE + 1;
export const CATERPILLAR_LINE_COUNT = 17;
export const caterpillarLength = CATERPILLAR_LINE_COUNT * CATERPILLAR_PADDING;

// Track anchor position (distance from tank center to track center)
export const TRACK_ANCHOR_Y = PADDING * 5 + SIZE * 0.3;

// Caterpillar parts - local coordinates relative to track entity (centered at 0,0)
export const caterpillarSet = createRectangleSet(
    CATERPILLAR_LINE_COUNT, 2,
    CATERPILLAR_SIZE, CATERPILLAR_PADDING,
    SIZE, PADDING,
);

// Left track caterpillar parts (local to left track)
export const caterpillarSetLeft: PartsData[] = caterpillarSet;

// Right track caterpillar parts (local to right track)
export const caterpillarSetRight: PartsData[] = caterpillarSet;

export const PARTS_COUNT = hullSet.length + turretHeadSet.length + turretGunSet.length + CATERPILLAR_LINE_COUNT * 2 * 2;

