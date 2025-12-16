import { addComponent, EntityId, World } from 'bitecs';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';

export const MAX_BULLET_SPEED = 400;
export const MIN_BULLET_SPEED = 80;

export enum BulletCaliber {
    Light,
    Medium,
    Heavy,
}

const BulletLightCaliber = {
    width: 3,
    height: 8,
    speed: 300,
    density: 3_000,
    damage: 3,
    linearDamping: 0.4, // Light bullets lose speed quickly
};
const BulletMediumCaliber = {
    width: 5,
    height: 10,
    speed: 350,
    density: 6_000,
    damage: 6,
    linearDamping: 0.2, // Medium bullets have moderate drag
};
const BulletHeavyCaliber = {
    width: 7,
    height: 14,
    speed: 450,
    density: 10_000,
    damage: 10,
    linearDamping: 0.075, // Heavy bullets maintain speed longer
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
