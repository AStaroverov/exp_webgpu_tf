import { delegate } from '../../delegate.ts';
import { NestedArray } from '../../utils.ts';

export const Shape = ({
    kind: new Uint8Array(delegate.defaultSize),
    values: NestedArray.f64(6, delegate.defaultSize),
});

export function setCircle(id: number, radius: number) {
    Shape.kind[id] = 0;
    Shape.values.set(id, 0, radius);
}

export function setRectangle(id: number, width: number, height: number) {
    Shape.kind[id] = 1;
    Shape.values.set(id, 0, width);
    Shape.values.set(id, 1, height);
}

export function setParallelogram(id: number, width: number, height: number, skew: number) {
    Shape.kind[id] = 3;
    Shape.values.set(id, 0, width);
    Shape.values.set(id, 1, height);
    Shape.values.set(id, 2, skew);
}

export function setTrapezoid(id: number, topWidth: number, bottomWidth: number, height: number) {
    Shape.kind[id] = 4;
    Shape.values.set(id, 0, topWidth);
    Shape.values.set(id, 1, bottomWidth);
    Shape.values.set(id, 2, height);
}

export function setTriangle(id: number, a: number, b: number, c: number, d: number, e: number, f: number) {
    Shape.kind[id] = 5;
    Shape.values.set(id, 0, a);
    Shape.values.set(id, 1, b);
    Shape.values.set(id, 2, c);
    Shape.values.set(id, 3, d);
    Shape.values.set(id, 4, e);
    Shape.values.set(id, 5, f);
}