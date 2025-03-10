import { TypedArray } from '../../../../../src/utils.ts';
import { delegate } from '../../../../../src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { component } from '../../../../../src/ECS/utils.ts';

export const Destroy = component({
    recursive: TypedArray.i8(delegate.defaultSize),

    addComponent(world: World, eid: number, recursive: boolean = true) {
        addComponent(world, eid, Destroy);
        Destroy.recursive[eid] = recursive ? 1 : 0;
    },
});

export const DestroyByTimeout = component({
    timeout: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: number, timeout: number) {
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