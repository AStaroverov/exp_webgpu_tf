import { delegate } from '../../delegate.ts';
import { component, obs, TypedArray } from '../../utils.ts';
import { addComponent, World } from 'bitecs';

export const Thinness = component({
    value: TypedArray.f64(delegate.defaultSize),

    addComponent: (world: World, eid: number, value = 0) => {
        addComponent(world, eid, Thinness);
        Thinness.value[eid] = value;
    },
    set$: obs((eid: number, value: number) => {
        Thinness.value[eid] = value;
    }),
});

export const Roundness = component({
    value: new Float64Array(delegate.defaultSize),

    addComponent: (world: World, eid: number, value = 0) => {
        addComponent(world, eid, Roundness);
        Roundness.value[eid] = value;
    },
    set$: obs((eid: number, value: number) => {
        Roundness.value[eid] = value;
    }),
});

export type TColor = [number, number, number, number] | Float32Array;
export const Color = component({
    r: new Float64Array(delegate.defaultSize),
    g: new Float64Array(delegate.defaultSize),
    b: new Float64Array(delegate.defaultSize),
    a: new Float64Array(delegate.defaultSize),

    addComponent(world: World, eid: number, r = 0, g = 0, b = 0, a = 1) {
        addComponent(world, eid, Color);
        Color.r[eid] = r;
        Color.g[eid] = g;
        Color.b[eid] = b;
        Color.a[eid] = a;
    },
    set$: obs((eid: number, r: number, g: number, b: number, a: number) => {
        Color.r[eid] = r;
        Color.g[eid] = g;
        Color.b[eid] = b;
        Color.a[eid] = a;
    }),
});

export type TShadow = [fadeStart: number, fadeEnd: number] | Float32Array;
export const Shadow = component({
    fadeStart: new Float64Array(delegate.defaultSize),
    fadeEnd: new Float64Array(delegate.defaultSize),

    addComponent(world: World, eid: number, fadeStart = 0, fadeEnd = 0) {
        addComponent(world, eid, Shadow);
        Shadow.fadeStart[eid] = fadeStart;
        Shadow.fadeEnd[eid] = fadeEnd;
    },
    set$: obs((eid: number, fadeStart: number, fadeEnd: number) => {
        Shadow.fadeStart[eid] = fadeStart;
        Shadow.fadeEnd[eid] = fadeEnd;
    }),
});

