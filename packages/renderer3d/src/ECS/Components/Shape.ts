import { delegate } from "../../delegate.ts";
import { NestedArray } from "../../utils.ts";
import { addComponent, World } from "bitecs";
import { defineComponent } from "../utils.ts";

export enum ShapeKind {
  Circle = 0,
  Rectangle = 1,
  Parallelogram = 3,
  Trapezoid = 4,
  Triangle = 5,
  // The only non-extruded primitive: a true 3D sphere (length(p) - r in the
  // shader). values[0] = radius. Every other kind is a 2D footprint extruded by
  // the entity's Height; Circle extrudes to a cylinder.
  Sphere = 6,
}

export const createShapeComponent = defineComponent((Shape, { obs }) => {
  const kind = new Uint8Array(delegate.defaultSize);
  const values = NestedArray.f64(6, delegate.defaultSize);

  return {
    kind,
    values,

    addComponent(
      world: World,
      id: number,
      k: ShapeKind = ShapeKind.Circle,
      a = 0,
      b = 0,
      c = 0,
      d = 0,
      e = 0,
      f = 0,
    ) {
      addComponent(world, id, Shape);
      kind[id] = k;
      values.set(id, 0, a);
      values.set(id, 1, b);
      values.set(id, 2, c);
      values.set(id, 3, d);
      values.set(id, 4, e);
      values.set(id, 5, f);
    },
    setCircle$: obs((id: number, radius: number) => {
      kind[id] = ShapeKind.Circle;
      values.set(id, 0, radius);
    }),
    setSphere$: obs((id: number, radius: number) => {
      kind[id] = ShapeKind.Sphere;
      values.set(id, 0, radius);
    }),
    setRectangle$: obs((id: number, width: number, height: number) => {
      kind[id] = ShapeKind.Rectangle;
      values.set(id, 0, width);
      values.set(id, 1, height);
    }),
    setParallelogram$: obs((id: number, width: number, height: number, skew: number) => {
      kind[id] = ShapeKind.Parallelogram;
      values.set(id, 0, width);
      values.set(id, 1, height);
      values.set(id, 2, skew);
    }),
    setTrapezoid$: obs((id: number, topWidth: number, bottomWidth: number, height: number) => {
      kind[id] = ShapeKind.Trapezoid;
      values.set(id, 0, topWidth);
      values.set(id, 1, bottomWidth);
      values.set(id, 2, height);
    }),
    setTriangle$: obs(
      (id: number, a: number, b: number, c: number, d: number, e: number, f: number) => {
        kind[id] = ShapeKind.Triangle;
        values.set(id, 0, a);
        values.set(id, 1, b);
        values.set(id, 2, c);
        values.set(id, 3, d);
        values.set(id, 4, e);
        values.set(id, 5, f);
      },
    ),
  };
});
