import type { PhysicalWorld } from "./initPhysicalWorld.ts";
import { createBody } from "./createBody.ts";
import { createRigidBox, createRigidBall } from "./createRigid.ts";
import type { SpawnBodyOp } from "./opChannel.ts";

// WORKER half of the RigidShapes split (plan §6.1/§6.3). RigidShapes' main half now
// only builds the render entity + posts the op; this turns that op back into a real
// Rapier body + collider and returns its handle (pid).
//
// The op carries the body CENTER and half-extents/radius directly, so the geometry
// math is already done on main (it must match the render placement there). Here we
// just mirror createBody + createRigid{Box,Ball}. `groundBox` is a fixed thin box —
// identical construction to a fixed box; the distinct kind only preserves intent.
export function spawnBodyFromOp(physicalWorld: PhysicalWorld, op: SpawnBodyOp): number {
  const body = createBody(physicalWorld, {
    type: op.bodyType,
    x: op.position.x,
    y: op.position.y,
    z: op.position.z,
  });

  switch (op.kind) {
    case "box":
    case "groundBox":
      return createRigidBox(physicalWorld, body, op.halfExtents.x, op.halfExtents.y, op.halfExtents.z);
    case "sphere":
      return createRigidBall(physicalWorld, body, op.radius);
  }
}
