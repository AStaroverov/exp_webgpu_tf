import { createRectangle, createSphere } from "../../../../renderer/src/ECS/Entities/Shapes.ts";
import { addTransformComponents } from "../../../../renderer/src/ECS/Components/Transform.ts";
import type { TColor } from "../../../../renderer/src/ECS/Components/Common.ts";
import {
  createEntityId,
  getEngineComponents,
  type EngineWorld,
} from "../../../../engine/src/ECS/createEngineWorld.ts";

function rand(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

export function buildTree(world: EngineWorld): number {
  const { Children } = getEngineComponents(world);

  const root = createEntityId(world);
  addTransformComponents(world, root);
  Children.addComponent(world, root);

  const trunkHeight = rand(3.5, 5);
  const trunkWidth = rand(0.4, 0.6);
  const trunkColor: TColor = [rand(0.38, 0.46), rand(0.26, 0.32), rand(0.14, 0.2), 1];
  const trunk = createRectangle(world, {
    eid: createEntityId(world),
    x: 0,
    y: 0,
    z: trunkHeight / 2,
    width: trunkWidth,
    height: trunkWidth,
    depth: trunkHeight,
    color: trunkColor,
  });
  Children.addChild(root, trunk);

  const canopyBase = trunkHeight;
  const blobs = Math.floor(rand(7, 11));
  for (let i = 0; i < blobs; i++) {
    const angle = rand(0, Math.PI * 2);
    const spread = rand(0, 1.4);
    const color: TColor = [rand(0.12, 0.26), rand(0.46, 0.7), rand(0.16, 0.28), 1];
    const blob = createSphere(world, {
      x: Math.cos(angle) * spread,
      y: Math.sin(angle) * spread,
      z: canopyBase + rand(0, 1.6),
      radius: rand(0.9, 1.5),
      color,
      eid: createEntityId(world),
    });
    Children.addChild(root, blob);
  }

  return root;
}
