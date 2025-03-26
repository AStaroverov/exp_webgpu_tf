import { addComponent } from 'bitecs';
import { delegate } from '../../../../../src/delegate.ts';
import { TypedArray } from '../../../../../src/utils.ts';
import { GameDI } from '../../DI/GameDI.ts';
import { component, obs } from '../../../../../src/ECS/utils.ts';

export const Hitable = component(({
    damage: TypedArray.f64(delegate.defaultSize),

    addComponent: (eid: number) => {
        addComponent(GameDI.world, eid, Hitable);
    },
    hit$: obs((eid: number, damage: number) => {
        Hitable.damage[eid] += damage;
    }),
}));
