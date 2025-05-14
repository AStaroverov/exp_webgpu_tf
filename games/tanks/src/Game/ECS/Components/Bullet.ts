import { addComponent, EntityId, World } from 'bitecs';

export const BULLET_SPEED = 400;

export const Bullet = {
    addComponent(world: World, eid: EntityId) {
        addComponent(world, eid, Bullet);
    },
};
