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

let animTimer: ReturnType<typeof setInterval> | undefined;

export function buildUnit(world: EngineWorld): number {
  if (animTimer !== undefined) clearInterval(animTimer);
  const { Children, LocalTransform } = getEngineComponents(world);

  const root = createEntityId(world);
  addTransformComponents(world, root);
  applyMatrixTranslate(LocalTransform.matrix.getBatch(root), 0, 0, HOVER);
  Children.addComponent(world, root);

  const add = (eid: number) => Children.addChild(root, eid);

  const bodyDepth = 2.4;
  let body = createTrapezoid(world, {
    x: 0,
    y: 0,
    z: bodyDepth / 2,
    topWidth: 1.7,
    bottomWidth: 0.6,
    height: 1.1,
    depth: bodyDepth,
    color: COLOR,
    eid: createEntityId(world),
  });
  add(body);

  const armSize = 0.45;
  const armZ = 1.5;
  add(
    createRectangle(world, {
      x: -1.0,
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
      x: 1.0,
      y: 0,
      z: armZ,
      width: armSize,
      height: armSize,
      depth: armSize,
      color: COLOR,
      eid: createEntityId(world),
    }),
  );

  const headSize = 0.7;
  add(
    createRectangle(world, {
      x: 0,
      y: 0,
      z: bodyDepth + 0.5 + headSize / 2,
      width: headSize,
      height: headSize,
      depth: headSize,
      color: COLOR,
      eid: createEntityId(world),
    }),
  );

  const bodyPos: [number, number, number] = [0, 0, bodyDepth / 2];
  let elapsed = 0;
  animTimer = setInterval(() => {
    elapsed += 0.033;
    const axis = Math.floor(elapsed / 3) % 3;
    const angle = ((elapsed % 3) / 3) * Math.PI * 2;
    const m = LocalTransform.matrix.getBatch(body);
    mat4.identity(m);
    mat4.translate(m, m, bodyPos);
    if (axis === 0) mat4.rotateX(m, m, angle);
    else if (axis === 1) mat4.rotateY(m, m, angle);
    else mat4.rotateZ(m, m, angle);
  }, 33);

  return root;
}
