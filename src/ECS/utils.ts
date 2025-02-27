import { addComponent } from 'bitecs';
import { DI } from '../../games/tanks/src/DI';

const $CompRef = Symbol('CompRef');
let indexCompRef = 0;
let nextCompRef: any = { [$CompRef]: indexCompRef++ };

export function component<T>(_comp: T): T {
    const comp = Object.assign(nextCompRef, _comp);
    nextCompRef = { [$CompRef]: indexCompRef++ };
    return comp;
}

export function obs<T extends (eid: number, ...args: A) => void, A extends any[]>(setter: T): T {
    const setData = { component: nextCompRef, data: null };
    return ((eid: number, ...args: A) => {
        const r = setter(eid, ...args);
        addComponent(DI.world, eid, setData);
        return r;
    }) as T;
}
