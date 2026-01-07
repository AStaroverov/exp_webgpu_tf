import { createRectangleSet, PartsData } from '../Common/TankParts.ts';

export const DENSITY = 250;
export const SIZE = 5;
export const PADDING = SIZE + 1;

export const hullSet = createRectangleSet(10, 8, SIZE, PADDING);
export const turretHeadSet = createRectangleSet(6, 5, SIZE, PADDING);
export const turretGunSet = createRectangleSet(6, 2, SIZE, PADDING).map((set) => {
    set[0] += (PADDING * 6);
    return set;
});

export const CATERPILLAR_LINE_COUNT = 12;
export const caterpillarLength = CATERPILLAR_LINE_COUNT * (PADDING - 1);

// Track anchor position (distance from tank center to track center)
export const TRACK_ANCHOR_Y = PADDING * 4 + SIZE;

// Caterpillar parts - local coordinates relative to track entity (centered at 0,0)
export const caterpillarSet = createRectangleSet(
    CATERPILLAR_LINE_COUNT, 1,
    SIZE - 1, PADDING - 1,
    SIZE + 2, PADDING + 2,
);

// Left track caterpillar parts (local to left track)
export const caterpillarSetLeft: PartsData[] = caterpillarSet;

// Right track caterpillar parts (local to right track)
export const caterpillarSetRight: PartsData[] = caterpillarSet;

export const PARTS_COUNT = hullSet.length + turretHeadSet.length + turretGunSet.length + CATERPILLAR_LINE_COUNT * 2;
