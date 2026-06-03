import { NestedArray, TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export const createDestroyComponent = defineComponent((Destroy) => {
    const recursive = TypedArray.i8(delegate.defaultSize);
    return {
        recursive,
        addComponent(world: World, eid: number, rec: boolean = true) {
            addComponent(world, eid, Destroy);
            recursive[eid] = rec ? 1 : 0;
        },
    };
});

export const createDestroyByTimeoutComponent = defineComponent((DestroyByTimeout) => {
    const timeout = TypedArray.f64(delegate.defaultSize);
    return {
        timeout,
        addComponent(world: World, eid: number, t: number) {
            addComponent(world, eid, DestroyByTimeout);
            timeout[eid] = t;
        },
        updateTimeout(eid: number, delta: number) {
            timeout[eid] -= delta;
        },
        resetTimeout(eid: number, t: number) {
            timeout[eid] = t;
        },
    };
});

export const createDestroyByDistanceComponent = defineComponent((DestroyByDistance) => {
    // Origin point the distance is measured from.
    const origin = NestedArray.f64(2, delegate.defaultSize);
    // Squared max distance — avoids a sqrt per entity per tick.
    const maxDistanceSq = TypedArray.f64(delegate.defaultSize);
    return {
        origin,
        maxDistanceSq,
        addComponent(world: World, eid: number, x: number, y: number, maxDistance: number) {
            addComponent(world, eid, DestroyByDistance);
            origin.set(eid, 0, x);
            origin.set(eid, 1, y);
            maxDistanceSq[eid] = maxDistance * maxDistance;
        },
    };
});
