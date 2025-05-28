import { createRectangleSet, PartsData } from '../Common/TankParts.ts';

export const DENSITY = 300;
export const SIZE = 6;
export const PADDING = SIZE + 1;

export const hullSet = createRectangleSet(8, 11, SIZE, PADDING);
export const turretHeadSet = createRectangleSet(6, 7, SIZE, PADDING);
export const turretGunSet = createRectangleSet(2, 10, SIZE, PADDING).map((set) => {
    set[1] -= (PADDING * 9 - SIZE / 2);
    return set;
});

export const CATERPILLAR_SIZE = 3;
export const CATERPILLAR_PADDING = CATERPILLAR_SIZE + 1;
export const CATERPILLAR_LINE_COUNT = 22;
export const caterpillarLength = CATERPILLAR_LINE_COUNT * CATERPILLAR_PADDING;
export const caterpillarSet = createRectangleSet(
    2, CATERPILLAR_LINE_COUNT,
    SIZE, PADDING,
    CATERPILLAR_SIZE, CATERPILLAR_PADDING,
);
export const caterpillarSetLeft: PartsData[] = caterpillarSet.map((set) => {
    set = set.slice() as PartsData;
    set[0] += PADDING * 5 + SIZE * 0.3;
    return set;
});
export const caterpillarSetRight: PartsData[] = caterpillarSet.map((set) => {
    set = set.slice() as PartsData;
    set[0] -= PADDING * 5 + SIZE * 0.3;
    return set;
});

export const PARTS_COUNT = hullSet.length + turretHeadSet.length + turretGunSet.length + CATERPILLAR_LINE_COUNT * 2;

