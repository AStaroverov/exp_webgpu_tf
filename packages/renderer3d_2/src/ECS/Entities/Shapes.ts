import { addEntity } from "bitecs";
import { addTransformComponents, applyMatrixTranslate } from "../Components/Transform.ts";
import { TColor } from "../Components/Common.ts";
import { ShapeKind } from "../Components/Shape.ts";
import { getRenderComponents, type RenderWorldLike } from "../world.ts";
import { adoptEntity } from "../../sab/adoptEntity.ts";

// Entity id source. When `eid` is supplied (the engine's shared-counter path,
// plan §4.2/§6.3) the world ADOPTS that exact id; otherwise bitecs allocates one
// from its own per-world counter (single-world renderer-only callers). Keeping
// this one helper means every shape factory honors a caller-chosen eid for free.
function createOrAdoptEntity(world: RenderWorldLike, eid?: number): number {
  return eid === undefined ? addEntity(world) : adoptEntity(world, eid);
}

// A sphere is the only true-3D primitive: its radius alone defines its vertical
// extent, so it takes no `height`. baseZ (`z`) is still the bottom of the shape.
export function createSphere(
  world: RenderWorldLike,
  {
    x,
    y,
    z,
    radius,
    color,
    eid,
  }: {
    x: number;
    y: number;
    z: number;
    radius: number;
    color: TColor;
    eid?: number;
  },
) {
  const { Color, LocalTransform, Shape, Height } = getRenderComponents(world);
  const id = createOrAdoptEntity(world, eid);

  addTransformComponents(world, id);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(id), x, y, z);

  Shape.addComponent(world, id, ShapeKind.Sphere, radius);
  Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
  // A sphere's diameter is its honest vertical extent.
  Height.addComponent(world, id, radius * 2);

  return id;
}

// A circle extrudes to a cylinder of the given `height` (default 0 = flat disc).
export function createCircle(
  world: RenderWorldLike,
  {
    x,
    y,
    z,
    radius,
    color,
    height,
  }: {
    x: number;
    y: number;
    z: number;
    radius: number;
    color: TColor;
    height?: number;
  },
) {
  const { Color, LocalTransform, Shape, Height } = getRenderComponents(world);
  const id = addEntity(world);

  addTransformComponents(world, id);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(id), x, y, z);

  Shape.addComponent(world, id, ShapeKind.Circle, radius);
  Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
  Height.addComponent(world, id, height ?? 0);

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
    depth,
    eid,
  }: {
    x: number;
    y: number;
    z: number;
    width: number;
    height: number;
    color: TColor;
    roundness?: number;
    // Vertical extent of the extruded box (default 0 = flat slab footprint).
    depth?: number;
    eid?: number;
  },
) {
  const { Color, LocalTransform, Roundness, Shape, Height } = getRenderComponents(world);
  const id = createOrAdoptEntity(world, eid);

  addTransformComponents(world, id);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(id), x, y, z);

  Shape.addComponent(world, id, ShapeKind.Rectangle, width, height);
  Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
  Roundness.addComponent(world, id, roundness ?? 0);
  Height.addComponent(world, id, depth ?? 0);

  return id;
}

export function createParallelogram(
  world: RenderWorldLike,
  {
    x,
    y,
    z,
    width,
    height,
    skew,
    color,
    roundness,
    depth,
  }: {
    x: number;
    y: number;
    z: number;
    width: number;
    height: number;
    skew: number;
    color: TColor;
    roundness?: number;
    depth?: number;
  },
) {
  const { Color, LocalTransform, Roundness, Shape, Height } = getRenderComponents(world);
  const id = addEntity(world);

  addTransformComponents(world, id);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(id), x, y, z);

  Shape.addComponent(world, id, ShapeKind.Parallelogram, width, height, skew);
  Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
  Roundness.addComponent(world, id, roundness ?? 0);
  Height.addComponent(world, id, depth ?? 0);

  return id;
}

export function createTrapezoid(
  world: RenderWorldLike,
  {
    x,
    y,
    z,
    topWidth,
    bottomWidth,
    height,
    color,
    roundness,
    depth,
  }: {
    x: number;
    y: number;
    z: number;
    topWidth: number;
    bottomWidth: number;
    height: number;
    color: TColor;
    roundness?: number;
    depth?: number;
  },
) {
  const { Color, LocalTransform, Roundness, Shape, Height } = getRenderComponents(world);
  const id = addEntity(world);

  addTransformComponents(world, id);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(id), x, y, z);

  Shape.addComponent(world, id, ShapeKind.Trapezoid, topWidth, bottomWidth, height);
  Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
  Roundness.addComponent(world, id, roundness ?? 0);
  Height.addComponent(world, id, depth ?? 0);

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
    depth,
  }: {
    x: number;
    y: number;
    z: number;
    roundness?: number;
    point1: [number, number];
    point2: [number, number];
    point3: [number, number];
    color: TColor;
    depth?: number;
  },
) {
  const { Color, LocalTransform, Roundness, Shape, Height } = getRenderComponents(world);
  const id = addEntity(world);

  addTransformComponents(world, id);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(id), x, y, z);

  Shape.addComponent(
    world,
    id,
    ShapeKind.Triangle,
    point1[0],
    point1[1],
    point2[0],
    point2[1],
    point3[0],
    point3[1],
  );
  Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
  Roundness.addComponent(world, id, roundness ?? 0);
  Height.addComponent(world, id, depth ?? 0);

  return id;
}
