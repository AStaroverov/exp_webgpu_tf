import { mat4 } from "gl-matrix";
import { createRectangle, createTrapezoid } from "../../../../renderer/src/ECS/Entities/Shapes.ts";
import {
  addTransformComponents,
  applyMatrixTranslate,
} from "../../../../renderer/src/ECS/Components/Transform.ts";
import type { TColor } from "../../../../renderer/src/ECS/Components/Common.ts";
import {
  createEntityId,
  getEngineComponents,
  type EngineWorld,
} from "../../../../engine/src/ECS/createEngineWorld.ts";

const COLOR: TColor = [0.13, 0.34, 0.56, 1];
const HOVER = 0.7;

// A unit (dark blue, one root + children): a downward-tapering pyramid torso (no legs),
// two small cube arms beside it, and a head cube floating a short gap above the torso. The
// torso is a trapezoid whose taper is horizontal in its own footprint; rotating it 90° about
// X stands the taper up so it is wide at the top and narrow at the bottom.
export function buildUnit(world: EngineWorld): number {
  const { Children, LocalTransform } = getEngineComponents(world);

  const root = createEntityId(world);
  addTransformComponents(world, root);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(root), 0, 0, HOVER);
  Children.addComponent(world, root);

  const add = (eid: number) => Children.addChild(root, eid);

  const bodyHeight = 2.2;
  const body = createTrapezoid(world, {
    x: 0,
    y: 0,
    z: bodyHeight / 2,
    topWidth: 1.4,
    bottomWidth: 0.4,
    height: bodyHeight,
    depth: 1.0,
    color: COLOR,
    eid: createEntityId(world),
  });
  const bodyMatrix = LocalTransform.matrix.getBatch(body);
  mat4.rotateX(bodyMatrix, bodyMatrix, Math.PI / 2);
  add(body);

  const armSize = 0.35;
  const armZ = bodyHeight * 0.45;
  add(
    createRectangle(world, {
      x: -0.9,
      y: 0,
      z: armZ,
      width: armSize,
      height: armSize,
      depth: armSize,
      color: COLOR,
      eid: createEntityId(world),
    }),
  );
  add(
    createRectangle(world, {
      x: 0.9,
      y: 0,
      z: armZ,
      width: armSize,
      height: armSize,
      depth: armSize,
      color: COLOR,
      eid: createEntityId(world),
    }),
  );

  const headSize = 0.6;
  const headGap = 0.3;
  add(
    createRectangle(world, {
      x: 0,
      y: 0,
      z: bodyHeight + headGap + headSize / 2,
      width: headSize,
      height: headSize,
      depth: headSize,
      color: COLOR,
      eid: createEntityId(world),
    }),
  );

  return root;
}
