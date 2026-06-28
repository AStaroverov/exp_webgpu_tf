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
  Sphere = 6,
}

// Per-kind values layout (stride 8, slots 0..7; unused slots stay 0). The Z extent
// of every kind lives here too — there is no separate Height component.
//   Circle/cylinder (0): [0]=radius, [1]=height(depth)
//   Rectangle/box   (1): [0]=width,  [1]=height, [2]=depth
//   Parallelogram   (3): [0]=width,  [1]=height, [2]=skew,  [3]=depth
//   Trapezoid       (4): [0]=topW,   [1]=botW,   [2]=ySize, [3]=depth
//   Triangle        (5): [0..5]=verts(ax,ay,bx,by,cx,cy),   [6]=depth
//   Sphere          (6): [0]=radius (the radius is its own full Z extent)
export const SHAPE_VALUES_STRIDE = 8;

export const createShapeComponent = defineComponent((Shape, { obs }) => {
  const kind = new Uint8Array(delegate.defaultSize);
  const values = NestedArray.f64(SHAPE_VALUES_STRIDE, delegate.defaultSize);

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
      g = 0,
      h = 0,
    ) {
      addComponent(world, id, Shape);
      kind[id] = k;
      values.set(id, 0, a);
      values.set(id, 1, b);
      values.set(id, 2, c);
      values.set(id, 3, d);
      values.set(id, 4, e);
      values.set(id, 5, f);
      values.set(id, 6, g);
      values.set(id, 7, h);
    },
    setCircle$: obs((id: number, radius: number, height = 0) => {
      kind[id] = ShapeKind.Circle;
      values.set(id, 0, radius);
      values.set(id, 1, height);
    }),
    setSphere$: obs((id: number, radius: number) => {
      kind[id] = ShapeKind.Sphere;
      values.set(id, 0, radius);
    }),
    setRectangle$: obs((id: number, width: number, height: number, depth = 0) => {
      kind[id] = ShapeKind.Rectangle;
      values.set(id, 0, width);
      values.set(id, 1, height);
      values.set(id, 2, depth);
    }),
    setParallelogram$: obs(
      (id: number, width: number, height: number, skew: number, depth = 0) => {
        kind[id] = ShapeKind.Parallelogram;
        values.set(id, 0, width);
        values.set(id, 1, height);
        values.set(id, 2, skew);
        values.set(id, 3, depth);
      },
    ),
    setTrapezoid$: obs(
      (id: number, topWidth: number, bottomWidth: number, height: number, depth = 0) => {
        kind[id] = ShapeKind.Trapezoid;
        values.set(id, 0, topWidth);
        values.set(id, 1, bottomWidth);
        values.set(id, 2, height);
        values.set(id, 3, depth);
      },
    ),
    setTriangle$: obs(
      (
        id: number,
        a: number,
        b: number,
        c: number,
        d: number,
        e: number,
        f: number,
        depth = 0,
      ) => {
        kind[id] = ShapeKind.Triangle;
        values.set(id, 0, a);
        values.set(id, 1, b);
        values.set(id, 2, c);
        values.set(id, 3, d);
        values.set(id, 4, e);
        values.set(id, 5, f);
        values.set(id, 6, depth);
      },
    ),
  };
});
