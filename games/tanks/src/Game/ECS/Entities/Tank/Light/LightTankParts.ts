import { Options } from '../Common/Options.ts';
import * as TankParts from '../Common/TankParts.ts';
import { createRectangleSet, PartsData } from '../Common/TankParts.ts';
import { BulletCaliber } from '../../../Components/Bullet.ts';

export const DENSITY = 250;
export const SIZE = 5;
export const PADDING = SIZE + 1;

const hullSet = createRectangleSet(8, 10, SIZE, PADDING);
const turretHeadSet = createRectangleSet(5, 6, SIZE, PADDING);
const turretGunSet = createRectangleSet(2, 6, SIZE, PADDING).map((set) => {
    set[1] -= (PADDING * 6);
    return set;
});

export const CATERPILLAR_LINE_COUNT = 12;
const caterpillarLength = CATERPILLAR_LINE_COUNT * (PADDING - 1);
const caterpillarSet = createRectangleSet(
    1, CATERPILLAR_LINE_COUNT,
    SIZE + 2, PADDING + 2,
    SIZE - 1, PADDING - 1,
);
const caterpillarSetLeft: PartsData[] = caterpillarSet.map((set) => {
    set = set.slice() as PartsData;
    set[0] += PADDING * 4 + SIZE;
    return set;
});
const caterpillarSetRight: PartsData[] = caterpillarSet.map((set) => {
    set = set.slice() as PartsData;
    set[0] -= PADDING * 4 + SIZE;
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
            position: [0, -9 * PADDING],
            caliber: BulletCaliber.Light,
        },
    }, options);
}
