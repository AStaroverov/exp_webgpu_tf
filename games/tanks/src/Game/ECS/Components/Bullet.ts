import { addComponent, EntityId, World } from 'bitecs';

export const MAX_BULLET_SPEED = 400;

export enum BulletCaliber {
    Light,
    Medium,
}

const BulletLightCaliber = {
    width: 3,
    height: 5,
    speed: 280,
    density: 8_000,
};
const BulletMediumCaliber = {
    width: 5,
    height: 7,
    speed: 300,
    density: 10_000,
};

export const mapBulletCaliber = {
    [BulletCaliber.Light]: BulletLightCaliber,
    [BulletCaliber.Medium]: BulletMediumCaliber,
};

export const Bullet = {
    addComponent(world: World, eid: EntityId) {
        addComponent(world, eid, Bullet);
    },
};
