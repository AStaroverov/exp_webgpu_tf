import { createRectangleSet, PartsData } from '../Common/TankParts.ts';

export const DENSITY = 250;
export const SIZE = 5;
export const PADDING = SIZE + 1;

export const hullSet = createRectangleSet(8, 10, SIZE, PADDING);
export const turretHeadSet = createRectangleSet(5, 6, SIZE, PADDING);
export const turretGunSet = createRectangleSet(2, 6, SIZE, PADDING).map((set) => {
    set[1] -= (PADDING * 6);
    return set;
});

export const CATERPILLAR_LINE_COUNT = 12;
export const caterpillarLength = CATERPILLAR_LINE_COUNT * (PADDING - 1);

// Track anchor position (distance from tank center to track center)
export const TRACK_ANCHOR_X = PADDING * 4 + SIZE;

// Caterpillar parts - local coordinates relative to track entity (centered at 0,0)
export const caterpillarSet = createRectangleSet(
    1, CATERPILLAR_LINE_COUNT,
    SIZE + 2, PADDING + 2,
    SIZE - 1, PADDING - 1,
);

// Left track caterpillar parts (local to left track)
export const caterpillarSetLeft: PartsData[] = caterpillarSet.map((set) => {
    return set.slice() as PartsData; // Already centered at track origin
});

// Right track caterpillar parts (local to right track)
export const caterpillarSetRight: PartsData[] = caterpillarSet.map((set) => {
    return set.slice() as PartsData; // Already centered at track origin
});

export const PARTS_COUNT = hullSet.length + turretHeadSet.length + turretGunSet.length + CATERPILLAR_LINE_COUNT * 2;
