import { addComponent, EntityId, World } from 'bitecs';
import { TypedArray } from '../../../../../../src/utils.ts';
import { delegate } from '../../../../../../src/delegate.ts';

export const MAX_BULLET_SPEED = 400;

export enum BulletCaliber {
    Light,
    Medium,
}

const BulletLightCaliber = {
    width: 3,
    height: 5,
    speed: 250,
    density: 1_000,
    damage: 3,
};
const BulletMediumCaliber = {
    width: 6,
    height: 8,
    speed: 330,
    density: 2_000,
    damage: 6,
};

export const mapBulletCaliber = {
    [BulletCaliber.Light]: BulletLightCaliber,
    [BulletCaliber.Medium]: BulletMediumCaliber,
};

export const Bullet = {
    caliber: TypedArray.i8(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, calibre: BulletCaliber) {
        addComponent(world, eid, Bullet);
        Bullet.caliber[eid] = calibre;
    },
};
