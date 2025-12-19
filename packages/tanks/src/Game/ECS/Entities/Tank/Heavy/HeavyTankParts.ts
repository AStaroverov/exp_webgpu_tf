import { createRectangleSet, PartsData } from '../Common/TankParts.ts';

export const DENSITY = 350;
export const SIZE = 8;
export const PADDING = SIZE + 1;

export const hullSet = createRectangleSet(10, 14, SIZE, PADDING);
export const turretHeadSet = createRectangleSet(7, 9, SIZE, PADDING);
export const turretGunSet = createRectangleSet(2, 8, SIZE, PADDING).map((set) => {
    set[1] -= (PADDING * 9 - SIZE / 2);
    return set;
});

export const CATERPILLAR_SIZE = 5;
export const CATERPILLAR_PADDING = CATERPILLAR_SIZE + 1;
export const CATERPILLAR_LINE_COUNT = 22;
export const caterpillarLength = CATERPILLAR_LINE_COUNT * CATERPILLAR_PADDING;

// Track anchor position (distance from tank center to track center)
export const TRACK_ANCHOR_X = PADDING * 6 + SIZE * 0.3;

// Caterpillar parts - local coordinates relative to track entity (centered at 0,0)
export const caterpillarSet = createRectangleSet(
    2, CATERPILLAR_LINE_COUNT,
    SIZE, PADDING,
    CATERPILLAR_SIZE, CATERPILLAR_PADDING,
);

// Left track caterpillar parts (local to left track)
export const caterpillarSetLeft: PartsData[] = caterpillarSet.map((set) => {
    return set.slice() as PartsData; // Already centered at track origin
});

// Right track caterpillar parts (local to right track)
export const caterpillarSetRight: PartsData[] = caterpillarSet.map((set) => {
    return set.slice() as PartsData; // Already centered at track origin
});

export const PARTS_COUNT = hullSet.length + turretHeadSet.length + turretGunSet.length + CATERPILLAR_LINE_COUNT * 2 * 2;

