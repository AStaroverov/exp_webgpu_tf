import { addComponent, World } from 'bitecs';
import { delegate } from '../../../../../src/delegate.ts';
import { TypedArray } from '../../../../../src/utils.ts';
import { component, obs } from '../../../../../src/ECS/utils.ts';

export const Hitable = component(({
    damage: TypedArray.f64(delegate.defaultSize),

    addComponent: (world: World, eid: number) => {
        addComponent(world, eid, Hitable);
    },
    hit$: obs((eid: number, damage: number) => {
        Hitable.damage[eid] += damage;
    }),
}));
