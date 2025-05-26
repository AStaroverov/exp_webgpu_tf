import { addComponent, World } from 'bitecs';
import { delegate } from '../../../../../../src/delegate.ts';
import { TypedArray } from '../../../../../../src/utils.ts';
import { component } from '../../../../../../src/ECS/utils.ts';

export const Damagable = component(({
    damage: TypedArray.f64(delegate.defaultSize),

    addComponent: (world: World, eid: number, damage: number) => {
        addComponent(world, eid, Damagable);
        Damagable.damage[eid] = damage;
    },
}));
