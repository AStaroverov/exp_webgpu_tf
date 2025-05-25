import { Options } from '../Common/Options.ts';
import * as TankParts from '../Common/TankParts.ts';
import { createRectangleSet, PartsData } from '../Common/TankParts.ts';
import { BulletCaliber } from '../../../Components/Bullet.ts';


export const DENSITY = 300;
export const SIZE = 6;
export const PADDING = SIZE + 1;

const hullSet = createRectangleSet(8, 11, SIZE, PADDING);
const turretHeadSet = createRectangleSet(6, 7, SIZE, PADDING);
const turretGunSet = createRectangleSet(2, 10, SIZE, PADDING).map((set) => {
    set[1] -= (PADDING * 9 - SIZE / 2);
    return set;
});

export const CATERPILLAR_SIZE = 3;
export const CATERPILLAR_PADDING = CATERPILLAR_SIZE + 1;
export const CATERPILLAR_LINE_COUNT = 22;
const caterpillarLength = CATERPILLAR_LINE_COUNT * CATERPILLAR_PADDING;
const caterpillarSet = createRectangleSet(
    2, CATERPILLAR_LINE_COUNT,
    SIZE, PADDING,
    CATERPILLAR_SIZE, CATERPILLAR_PADDING,
);
const caterpillarSetLeft: PartsData[] = caterpillarSet.map((set) => {
    set = set.slice() as PartsData;
    set[0] += PADDING * 5 + SIZE * 0.3;
    return set;
});
const caterpillarSetRight: PartsData[] = caterpillarSet.map((set) => {
    set = set.slice() as PartsData;
    set[0] -= PADDING * 5 + SIZE * 0.3;
    return set;
});


export const PARTS_COUNT = hullSet.length + turretHeadSet.length + turretGunSet.length + CATERPILLAR_LINE_COUNT * 2;

export function createTankHullParts(tankEid: number, options: Options): void {
    options.size = SIZE;
    options.padding = PADDING;
    options.density = DENSITY * 10;
    return TankParts.createTankHullParts(tankEid, hullSet, options);
}

export function createTankCaterpillarsParts(tankEid: number, options: Options): void {
    options.size = SIZE;
    options.padding = PADDING;
    options.density = DENSITY;
    TankParts.createTankCaterpillarsParts(
        tankEid,
        [[caterpillarLength, caterpillarSetLeft], [caterpillarLength, caterpillarSetRight]],
        options,
    );
}

export function createTankTurretParts(turretEid: number, options: Options) {
    options.size = SIZE;
    options.padding = PADDING;
    options.density = DENSITY;
    return TankParts.createTankTurretParts(turretEid, {
        headSet: turretHeadSet, gunSet: turretGunSet,
        bullet: {
            position: [0, -13 * PADDING],
            caliber: BulletCaliber.Medium,
        },
    }, options);
}
