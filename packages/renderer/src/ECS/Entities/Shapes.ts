import { addEntity } from "bitecs";
import { addTransformComponents, applyMatrixTranslate } from "../Components/Transform.ts";
import { TColor } from "../Components/Common.ts";
import { ShapeKind } from "../Components/Shape.ts";
import { getRenderComponents, type RenderWorldLike } from "../world.ts";

// z half-thickness for flat 2D shapes promoted to 3D boxes.
const FLAT_HALF_THICKNESS = 1;

export function createCircle(
  world: RenderWorldLike,
  {
    x,
    y,
    z,
    radius,
    color,
  }: {
    x: number;
    y: number;
    z: number;
    radius: number;
    color: TColor;
  },
) {
  const { Color, LocalTransform, Shape } = getRenderComponents(world);
  const id = addEntity(world);

  addTransformComponents(world, id);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(id), x, y, z);

  Shape.addComponent(world, id, ShapeKind.Sphere3D, radius);
  Color.addComponent(world, id, color[0], color[1], color[2], color[3]);

  return id;
}

export function createRectangle(
  world: RenderWorldLike,
  {
    x,
    y,
    z,
    width,
    height,
    color,
    roundness,
  }: {
    x: number;
    y: number;
    z: number;
    width: number;
    height: number;
    color: TColor;
    roundness?: number;
  },
) {
  const { Color, LocalTransform, Roundness, Shape } = getRenderComponents(world);
  const id = addEntity(world);

  addTransformComponents(world, id);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(id), x, y, z);

  // Box3D stores half-extents; a flat rectangle gets a small z half-thickness
  // so it has volume for the impostor to trace.
  Shape.addComponent(world, id, ShapeKind.Box3D, width / 2, height / 2, FLAT_HALF_THICKNESS);
  Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
  Roundness.addComponent(world, id, roundness ?? 0);

  return id;
}

export function createTriangle(
  world: RenderWorldLike,
  {
    x,
    y,
    z,
    color,
    roundness,
    point1,
    point2,
    point3,
  }: {
    x: number;
    y: number;
    z: number;
    roundness?: number;
    point1: [number, number];
    point2: [number, number];
    point3: [number, number];
    color: TColor;
  },
) {
  const { Color, LocalTransform, Roundness, Shape } = getRenderComponents(world);
  const id = addEntity(world);

  addTransformComponents(world, id);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(id), x, y, z);

  // No 3D triangle primitive yet — approximate with its bounding box (Box3D
  // half-extents), centered on the local origin, with a small z thickness.
  const hx = Math.max(Math.abs(point1[0]), Math.abs(point2[0]), Math.abs(point3[0]));
  const hy = Math.max(Math.abs(point1[1]), Math.abs(point2[1]), Math.abs(point3[1]));
  Shape.addComponent(world, id, ShapeKind.Box3D, hx, hy, FLAT_HALF_THICKNESS);
  Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
  Roundness.addComponent(world, id, roundness ?? 0);

  return id;
}
