import { delegate } from '../../delegate.ts';
import { createMethods, TypedArray } from '../../utils.ts';
import { addComponent, World } from 'bitecs';

export const Thinness = ({
    value: TypedArray.f64(delegate.defaultSize),
});

export const ThinnessMethods = createMethods(Thinness, {
    addComponent(world: World, eid: number, value = 0) {
        addComponent(world, eid, Thinness);
        Thinness.value[eid] = value;
    },
    set$(eid: number, value: number) {
        Thinness.value[eid] = value;
    },
});

export const Roundness = ({
    value: new Float64Array(delegate.defaultSize),
});

export const RoundnessMethods = createMethods(Roundness, {
    addComponent(world: World, eid: number) {
        addComponent(world, eid, Roundness);
        Roundness.value[eid] = 0;
    },
    set$(eid: number, value: number) {
        Roundness.value[eid] = value;
    },
});

export type TColor = [number, number, number, number] | Float32Array;
export const Color = ({
    r: new Float64Array(delegate.defaultSize),
    g: new Float64Array(delegate.defaultSize),
    b: new Float64Array(delegate.defaultSize),
    a: new Float64Array(delegate.defaultSize),
});

export const ColorMethods = createMethods(Color, {
    addComponent(world: World, eid: number, r = 0, g = 0, b = 0, a = 1) {
        addComponent(world, eid, Color);
        Color.r[eid] = r;
        Color.g[eid] = g;
        Color.b[eid] = b;
        Color.a[eid] = a;
    },
    set$(eid: number, r: number, g: number, b: number, a: number) {
        Color.r[eid] = r;
        Color.g[eid] = g;
        Color.b[eid] = b;
        Color.a[eid] = a;
    },
});


export type TShadow = [fadeStart: number, fadeEnd: number] | Float32Array;
export const Shadow = ({
    fadeStart: new Float64Array(delegate.defaultSize),
    fadeEnd: new Float64Array(delegate.defaultSize),
});

export const ShadowMethods = createMethods(Shadow, {
    addComponent(world: World, eid: number, fadeStart = 0, fadeEnd = 0) {
        addComponent(world, eid, Shadow);
        Shadow.fadeStart[eid] = fadeStart;
        Shadow.fadeEnd[eid] = fadeEnd;
    },
    set$(eid: number, fadeStart: number, fadeEnd: number) {
        Shadow.fadeStart[eid] = fadeStart;
        Shadow.fadeEnd[eid] = fadeEnd;
    },
});

