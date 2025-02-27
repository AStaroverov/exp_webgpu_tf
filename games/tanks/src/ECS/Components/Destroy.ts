import { TypedArray } from '../../../../../src/utils.ts';
import { delegate } from '../../../../../src/delegate.ts';
import { addComponent } from 'bitecs';
import { DI } from '../../DI';
import { component } from '../../../../../src/ECS/utils.ts';

export const DestroyByTimeout = component({
    timeout: TypedArray.f64(delegate.defaultSize),

    addComponent(eid: number, timeout: number, { world } = DI) {
        addComponent(world, eid, DestroyByTimeout);
        DestroyByTimeout.timeout[eid] = timeout;
    },

    updateTimeout(eid: number, delta: number) {
        DestroyByTimeout.timeout[eid] -= delta;
    },
    resetTimeout(eid: number, timeout: number) {
        DestroyByTimeout.timeout[eid] = timeout;
    },
});