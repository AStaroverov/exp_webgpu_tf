import { defineComponent, Types } from 'bitecs';

export const Shape = defineComponent({
    kind: Types.ui32,
    values: [Types.f32, 6],
});

export function setCircle(id: number, radius: number) {
    Shape.kind[id] = 0;
    Shape.values[id][0] = radius;
}

export function setRectangle(id: number, width: number, height: number) {
    Shape.kind[id] = 1;
    Shape.values[id][0] = width;
    Shape.values[id][1] = height;
}

export function setParallelogram(id: number, width: number, height: number, skew: number) {
    Shape.kind[id] = 3;
    Shape.values[id][0] = width;
    Shape.values[id][1] = height;
    Shape.values[id][2] = skew;
}

export function setTrapezoid(id: number, topWidth: number, bottomWidth: number, height: number) {
    Shape.kind[id] = 4;
    Shape.values[id][0] = topWidth;
    Shape.values[id][1] = bottomWidth;
    Shape.values[id][2] = height;
}

export function setTriangle(id: number, a: number, b: number, c: number, d: number, e: number, f: number) {
    Shape.kind[id] = 5;
    Shape.values[id][0] = a;
    Shape.values[id][1] = b;
    Shape.values[id][2] = c;
    Shape.values[id][3] = d;
    Shape.values[id][4] = e;
    Shape.values[id][5] = f;
}