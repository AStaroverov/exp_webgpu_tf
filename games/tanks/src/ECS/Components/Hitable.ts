import { addComponent } from 'bitecs';
import { delegate } from '../../../../../src/delegate.ts';
import { component, obs, TypedArray } from '../../../../../src/utils.ts';
import { DI } from '../../DI';

export const Hitable = component(({
    damage: TypedArray.f64(delegate.defaultSize),

    addComponent: (eid: number) => {
        addComponent(DI.world, eid, Hitable);
    },
    hit$: obs((eid: number, damage: number) => {
        Hitable.damage[eid] += damage;
    }),
}));
