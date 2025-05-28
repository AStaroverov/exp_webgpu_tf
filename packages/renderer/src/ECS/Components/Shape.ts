import { delegate } from '../../delegate.ts';
import { NestedArray } from '../../utils.ts';
import { addComponent, World } from 'bitecs';
import { component, obs } from '../utils.ts';

export enum ShapeKind {
    Circle = 0,
    Rectangle = 1,
    Parallelogram = 3,
    Trapezoid = 4,
    Triangle = 5,
}

export const Shape = component({
    kind: new Uint8Array(delegate.defaultSize),
    values: NestedArray.f64(6, delegate.defaultSize),

    addComponent: (world: World, id: number, k = ShapeKind.Circle, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0) => {
        addComponent(world, id, Shape);
        Shape.kind[id] = k;
        Shape.values.set(id, 0, a);
        Shape.values.set(id, 1, b);
        Shape.values.set(id, 2, c);
        Shape.values.set(id, 3, d);
        Shape.values.set(id, 4, e);
        Shape.values.set(id, 5, f);
    },
    setCircle$: obs((id: number, radius: number) => {
        Shape.kind[id] = ShapeKind.Circle;
        Shape.values.set(id, 0, radius);
    }),
    setRectangle$: obs((id: number, width: number, height: number) => {
        Shape.kind[id] = ShapeKind.Rectangle;
        Shape.values.set(id, 0, width);
        Shape.values.set(id, 1, height);
    }),
    setParallelogram$: obs((id: number, width: number, height: number, skew: number) => {
        Shape.kind[id] = ShapeKind.Parallelogram;
        Shape.values.set(id, 0, width);
        Shape.values.set(id, 1, height);
        Shape.values.set(id, 2, skew);
    }),
    setTrapezoid$: obs((id: number, topWidth: number, bottomWidth: number, height: number) => {
        Shape.kind[id] = ShapeKind.Trapezoid;
        Shape.values.set(id, 0, topWidth);
        Shape.values.set(id, 1, bottomWidth);
        Shape.values.set(id, 2, height);
    }),
    setTriangle$: obs((id: number, a: number, b: number, c: number, d: number, e: number, f: number) => {
        Shape.kind[id] = ShapeKind.Triangle;
        Shape.values.set(id, 0, a);
        Shape.values.set(id, 1, b);
        Shape.values.set(id, 2, c);
        Shape.values.set(id, 3, d);
        Shape.values.set(id, 4, e);
        Shape.values.set(id, 5, f);
    }),
});
