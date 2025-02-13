import { addComponent, defineComponent, IWorld, Types } from 'bitecs';

export const Hitable = defineComponent({
    damage: Types.f64,
});

export function addHitableComponent(world: IWorld, entity: number) {
    addComponent(world, Hitable, entity);
    Hitable.damage[entity] = 0;
}

export function hit(entity: number, damage: number) {
    Hitable.damage[entity] += damage;
}
