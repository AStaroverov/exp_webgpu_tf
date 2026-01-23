import { addComponent, World } from 'bitecs';
import { delegate } from 'renderer/src/delegate.ts';
import { TypedArray } from 'renderer/src/utils.ts';
import { component } from 'renderer/src/ECS/utils.ts';

export const Damagable = component(({
    damage: TypedArray.f64(delegate.defaultSize),

    addComponent: (world: World, eid: number, damage: number) => {
        addComponent(world, eid, Damagable);
        Damagable.damage[eid] = damage;
    },
}));
