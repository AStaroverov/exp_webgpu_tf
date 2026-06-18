import { delegate } from "../../delegate.ts";
import { NestedArray } from "../../utils.ts";
import { addComponent, World } from "bitecs";
import { defineComponent } from "../utils.ts";

// 3D impostor primitives (the only kinds the SDF kernel understands). A flat
// tile is just a thin Box3D (one near-zero half-extent), so there is no separate
// plane primitive.
//
// uValues row layout (6 floats):
//   Box3D    : (hx, hy, hz)              half-extents
//   Sphere3D : (r)                       radius
export enum ShapeKind {
  Box3D = 10,
  Sphere3D = 11,
}

// Thickness given to flat 2D shapes promoted to 3D boxes (half-extent on z).
const FLAT_HALF_THICKNESS = 1;

export const createShapeComponent = defineComponent((Shape, { obs }) => {
  const kind = new Uint8Array(delegate.defaultSize);
  const values = NestedArray.f64(6, delegate.defaultSize);

  return {
    kind,
    values,

    addComponent(
      world: World,
      id: number,
      k: ShapeKind = ShapeKind.Sphere3D,
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

    // ── 3D setters ──────────────────────────────────────────────────────
    setBox$: obs((id: number, hx: number, hy: number, hz: number) => {
      kind[id] = ShapeKind.Box3D;
      values.set(id, 0, hx);
      values.set(id, 1, hy);
      values.set(id, 2, hz);
    }),
    setSphere$: obs((id: number, radius: number) => {
      kind[id] = ShapeKind.Sphere3D;
      values.set(id, 0, radius);
    }),

    // ── legacy 2D setters (promoted to 3D primitives) ───────────────────
    setCircle$: obs((id: number, radius: number) => {
      kind[id] = ShapeKind.Sphere3D;
      values.set(id, 0, radius);
    }),
    setRectangle$: obs((id: number, width: number, height: number) => {
      kind[id] = ShapeKind.Box3D;
      values.set(id, 0, width / 2);
      values.set(id, 1, height / 2);
      values.set(id, 2, FLAT_HALF_THICKNESS);
    }),
  };
});
