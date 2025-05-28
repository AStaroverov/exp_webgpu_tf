import { addComponent, EntityId, World } from 'bitecs';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';

export const MAX_BULLET_SPEED = 400;

export enum BulletCaliber {
    Light,
    Medium,
    Heavy,
}

const BulletLightCaliber = {
    width: 3,
    height: 5,
    speed: 300,
    density: 3_000,
    damage: 3,
};
const BulletMediumCaliber = {
    width: 5,
    height: 7,
    speed: 350,
    density: 6_000,
    damage: 6,
};
const BulletHeavyCaliber = {
    width: 7,
    height: 11,
    speed: 450,
    density: 10_000,
    damage: 10,
};

export const mapBulletCaliber = {
    [BulletCaliber.Light]: BulletLightCaliber,
    [BulletCaliber.Medium]: BulletMediumCaliber,
    [BulletCaliber.Heavy]: BulletHeavyCaliber,
};

export const Bullet = {
    caliber: TypedArray.i8(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, calibre: BulletCaliber) {
        addComponent(world, eid, Bullet);
        Bullet.caliber[eid] = calibre;
    },
};
