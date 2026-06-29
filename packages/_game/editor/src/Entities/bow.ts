import { mat4 } from "gl-matrix";
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

const WOOD: TColor = [0.42, 0.27, 0.12, 1];
const STRING: TColor = [0.85, 0.82, 0.7, 1];

// Bow standing string-up: two straight wooden limbs angling back from a central grip (belly toward
// -Y) to the tips at (0, ±H), and a straight string joining the tips along Z. Four shapes total.
const HALF_HEIGHT = 1.0;
const BELLY = 0.3;
const LIMB_RADIUS = 0.06;
const STRING_RADIUS = 0.02;
const GRIP_RADIUS = 0.1;
const GRIP_HEIGHT = 0.5;

export function buildBow(world: EngineWorld, { scale }: EntityOptions): EntityInstance {
  const { Children, LocalTransform } = getEngineComponents(world);
  const add = (eid: number) => Children.addChild(root, eid);

  const root = createEntityId(world);
  addTransformComponents(world, root);
  applyMatrixScale(LocalTransform.matrix.getBatch(root), scale);
  Children.addComponent(world, root);

  const halfGrip = GRIP_HEIGHT / 2;
  const limbDz = HALF_HEIGHT - halfGrip;
  const limbLen = Math.hypot(BELLY, limbDz);
  for (const sign of [1, -1]) {
    // Each limb spans from a grip end (-BELLY, H ± halfGrip) out to a tip (0, H ± H).
    const limb = createCircle(world, {
      x: 0,
      y: -BELLY / 2,
      z: HALF_HEIGHT + (sign * (HALF_HEIGHT + halfGrip)) / 2,
      radius: LIMB_RADIUS,
      height: limbLen,
      color: WOOD,
      eid: createEntityId(world),
    });
    const m = LocalTransform.matrix.getBatch(limb);
    mat4.rotateX(m, m, Math.atan2(-BELLY, sign * limbDz));
    add(limb);
  }

  add(
    createCircle(world, {
      x: 0,
      y: 0,
      z: HALF_HEIGHT,
      radius: STRING_RADIUS,
      height: 2 * HALF_HEIGHT,
      color: STRING,
      eid: createEntityId(world),
    }),
  );

  add(
    createCircle(world, {
      x: 0,
      y: -BELLY,
      z: HALF_HEIGHT,
      radius: GRIP_RADIUS,
      height: GRIP_HEIGHT,
      color: WOOD,
      eid: createEntityId(world),
    }),
  );

  return { root, bones: { root }, animations: {} };
}
