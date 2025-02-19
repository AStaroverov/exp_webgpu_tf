import { addComponent, World } from 'bitecs';
import { delegate } from '../../../../../src/delegate.ts';
import { createMethods, TypedArray } from '../../../../../src/utils.ts';

export const Hitable = ({
    damage: TypedArray.f64(delegate.defaultSize),
});

export const HitableMethods = createMethods(Hitable, {
    addComponent: (world: World, eid: number) => addComponent(world, eid, Hitable),
    hit$: (eid: number, damage: number) => {
        Hitable.damage[eid] += damage;
    },
});
