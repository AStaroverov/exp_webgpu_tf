import { defineComponent, Types } from 'bitecs';

export const Resolution = defineComponent({
    x: Types.f32,
    y: Types.f32,
});

export const Translate = defineComponent({
    x: Types.f32,
    y: Types.f32,
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

export function setColor(id: number, r: number, g: number, b: number, a: number) {
    Color.r[id] = r;
    Color.g[id] = g;
    Color.b[id] = b;
    Color.a[id] = a;
}