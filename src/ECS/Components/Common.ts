import { delegate } from '../../delegate.ts';

export const Enter = {};

export const Changed = {};

export const Size = ({
    width: new Float64Array(delegate.defaultSize),
    height: new Float64Array(delegate.defaultSize),
});

export const Thinness = ({
    value: new Float64Array(delegate.defaultSize),
});

export const Roundness = ({
    value: new Float64Array(delegate.defaultSize),
});

export const Color = ({
    r: new Float64Array(delegate.defaultSize),
    g: new Float64Array(delegate.defaultSize),
    b: new Float64Array(delegate.defaultSize),
    a: new Float64Array(delegate.defaultSize),
});

export type TColor = [number, number, number, number] | Float32Array;

export function setColor(id: number, r: number, g: number, b: number, a: number) {
    Color.r[id] = r;
    Color.g[id] = g;
    Color.b[id] = b;
    Color.a[id] = a;
}

export const Shadow = ({
    fadeStart: new Float64Array(delegate.defaultSize),
    fadeEnd: new Float64Array(delegate.defaultSize),
});

export type TShadow = [fadeStart: number, fadeEnd: number] | Float32Array;

export function setShadow(id: number, fadeStart: number, fadeEnd: number) {
    Shadow.fadeStart[id] = fadeStart;
    Shadow.fadeEnd[id] = fadeEnd;
}
