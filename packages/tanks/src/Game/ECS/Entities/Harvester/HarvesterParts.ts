import { createRectangleSet, PartsData } from '../Vehicle/VehicleParts.ts';

// Harvester is similar in size to medium tank (bulldozer-like)
export const DENSITY = 350;
export const SIZE = 6;
export const PADDING = SIZE + 1;

// Hull - wider and slightly shorter than medium tank (bulldozer shape)
export const hullSet = createRectangleSet(10, 9, SIZE, PADDING);

// Barrier "turret" - wide shield that rotates, no gun barrel
// Acts as impenetrable barrier instead of weapon
export const barrierSet = createRectangleSet(8, 4, SIZE, PADDING);

// Front scoop - curved shape for collecting debris
// Positioned at the front of the harvester
export const scoopSet: PartsData[] = createRectangleSet(12, 2, SIZE, PADDING).map((set) => {
    set[1] -= (PADDING * 6);
    return set;
});

// Caterpillar configuration - heavy duty tracks
export const CATERPILLAR_SIZE = 4;
export const CATERPILLAR_PADDING = CATERPILLAR_SIZE + 1;
export const CATERPILLAR_LINE_COUNT = 20;
export const caterpillarLength = CATERPILLAR_LINE_COUNT * CATERPILLAR_PADDING;
export const caterpillarSet = createRectangleSet(
    2, CATERPILLAR_LINE_COUNT,
    SIZE, PADDING,
    CATERPILLAR_SIZE, CATERPILLAR_PADDING,
);
export const caterpillarSetLeft: PartsData[] = caterpillarSet.map((set) => {
    set = set.slice() as PartsData;
    set[0] += PADDING * 6 + SIZE * 0.3;
    return set;
});
export const caterpillarSetRight: PartsData[] = caterpillarSet.map((set) => {
    set = set.slice() as PartsData;
    set[0] -= PADDING * 6 + SIZE * 0.3;
    return set;
});

export const PARTS_COUNT = hullSet.length + barrierSet.length + scoopSet.length + CATERPILLAR_LINE_COUNT * 2 * 2;
