import { defineComponent, Types } from 'bitecs';

export const Size = defineComponent({
    width: Types.f32,
    height: Types.f32,
});

export const Thinness = defineComponent({
    value: Types.f32,
});

export const Roundness = defineComponent({
    value: Types.f32,
});

export const Color = defineComponent({
    r: Types.f32,
    g: Types.f32,
    b: Types.f32,
    a: Types.f32,
});

export type TColor = [number, number, number, number] | Float32Array;

export function setColor(id: number, r: number, g: number, b: number, a: number) {
    Color.r[id] = r;
    Color.g[id] = g;
    Color.b[id] = b;
    Color.a[id] = a;
}

export const Shadow = defineComponent({
    fadeStart: Types.f32,
    fadeEnd: Types.f32,
});

export type TShadow = [fadeStart: number, fadeEnd: number] | Float32Array;

export function setShadow(id: number, fadeStart: number, fadeEnd: number) {
    Shadow.fadeStart[id] = fadeStart;
    Shadow.fadeEnd[id] = fadeEnd;
}