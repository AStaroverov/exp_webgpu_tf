import { createCircle } from "../../../../renderer/src/ECS/Entities/Shapes.ts";
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

const METAL: TColor = [0.55, 0.57, 0.62, 1];
const BLADE: TColor = [0.55, 0.8, 1.0, 1];

// A lightsaber standing blade-up: a metal hilt (pommel + grip) and a glowing energy blade.
// The blade is a thin bright cylinder that is also a LightEmitter, so it glows and lights the
// scene through the voxel GI.
export function buildLightsaber(world: EngineWorld, { scale }: EntityOptions): EntityInstance {
  const { Children, LocalTransform, LightEmitter } = getEngineComponents(world);

  const root = createEntityId(world);
  addTransformComponents(world, root);
  const rootMatrix = LocalTransform.matrix.getBatch(root);
  applyMatrixScale(rootMatrix, scale);
  Children.addComponent(world, root);

  const add = (eid: number) => Children.addChild(root, eid);

  add(
    createCircle(world, {
      x: 0,
      y: 0,
      z: 0.25,
      radius: 0.085,
      height: 0.5,
      color: METAL,
      eid: createEntityId(world),
    }),
  );

  const bladeLen = 1.8;
  const blade = createCircle(world, {
    x: 0,
    y: 0,
    z: 0.5 + bladeLen / 2,
    radius: 0.1,
    height: bladeLen,
    color: BLADE,
    eid: createEntityId(world),
  });
  LightEmitter.addComponent(world, blade, 5, 0.5);
  add(blade);

  return { root, animations: {} };
}
