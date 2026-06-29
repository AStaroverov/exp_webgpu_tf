import { mat4 } from "gl-matrix";
import {
  createCircle,
  createRectangle,
  createTriangle,
} from "../../../../renderer/src/ECS/Entities/Shapes.ts";
import {
  addTransformComponents,
  applyMatrixScale,
} from "../../../../renderer/src/ECS/Components/Transform.ts";
import type { TColor } from "../../../../renderer/src/ECS/Components/Common.ts";
import {
  createEntityId,
  getEngineComponents,
  type EngineWorld,
} from "../../../../engine/src/ECS/createEngineWorld.ts";
import type { EntityInstance, EntityOptions } from "./registry.ts";

const SHAFT: TColor = [0.5, 0.4, 0.28, 1];
const HEAD: TColor = [1.0, 0.13, 0.1, 1];
const FLETCH: TColor = [0.82, 0.82, 0.85, 1];

// Arrow standing point-up along Z: a thin shaft, a glowing-red triangular head at the top
// (a LightEmitter, so it lights the scene through the voxel GI), and two perpendicular fletching
// fins at the tail. The head triangle's footprint lives in XY, so it is rotated +90° about X to
// stand in the XZ plane with its apex pointing up.
const SHAFT_LEN = 2.2;
const SHAFT_RADIUS = 0.035;
const HEAD_LEN = 0.35;
const HEAD_HALF_WIDTH = 0.11;
const HEAD_THICKNESS = 0.08;
const FLETCH_LEN = 0.55;
const FLETCH_SPAN = 0.32;
const FLETCH_THICKNESS = 0.02;

export function buildArrow(world: EngineWorld, { scale }: EntityOptions): EntityInstance {
  const { Children, LocalTransform, LightEmitter } = getEngineComponents(world);

  const root = createEntityId(world);
  addTransformComponents(world, root);
  applyMatrixScale(LocalTransform.matrix.getBatch(root), scale);
  Children.addComponent(world, root);

  const add = (eid: number) => Children.addChild(root, eid);

  add(
    createCircle(world, {
      x: 0,
      y: 0,
      z: SHAFT_LEN / 2,
      radius: SHAFT_RADIUS,
      height: SHAFT_LEN,
      color: SHAFT,
      eid: createEntityId(world),
    }),
  );

  const head = createTriangle(world, {
    x: 0,
    y: 0,
    z: SHAFT_LEN,
    point1: [0, HEAD_LEN],
    point2: [-HEAD_HALF_WIDTH, 0],
    point3: [HEAD_HALF_WIDTH, 0],
    depth: HEAD_THICKNESS,
    color: HEAD,
  });
  const headMatrix = LocalTransform.matrix.getBatch(head);
  mat4.rotateX(headMatrix, headMatrix, Math.PI / 2);
  LightEmitter.addComponent(world, head, 5, 0.4);
  add(head);

  // Two perpendicular fins: one thin in Y (XZ plane), one thin in X (YZ plane).
  add(
    createRectangle(world, {
      x: 0,
      y: 0,
      z: FLETCH_LEN / 2,
      width: FLETCH_SPAN,
      height: FLETCH_THICKNESS,
      depth: FLETCH_LEN,
      color: FLETCH,
      eid: createEntityId(world),
    }),
  );
  add(
    createRectangle(world, {
      x: 0,
      y: 0,
      z: FLETCH_LEN / 2,
      width: FLETCH_THICKNESS,
      height: FLETCH_SPAN,
      depth: FLETCH_LEN,
      color: FLETCH,
      eid: createEntityId(world),
    }),
  );

  return { root, bones: { root }, animations: {} };
}
