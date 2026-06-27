import { RigidBodyDesc, type RigidBody, type Rotation } from "@dimforge/rapier3d-simd";
import type { PhysicalWorld } from "./initPhysicalWorld.ts";
import { IDENTITY_QUAT } from "../lib/constants.ts";

export type BodyType = "dynamic" | "fixed";

export type CreateBodyOptions = {
  type: BodyType;
  x: number;
  y: number;
  z: number;
  // Rapier Rotation interface {x,y,z,w}; identity = (0,0,0,1). Plain objects are
  // accepted. Defaults to no rotation.
  rot?: Rotation;
};

// Generic rigid-body builder: 3D translation (3 args) + a quaternion rotation.
// The collider is attached separately by createRigid* (so body and shape stay
// decoupled).
export function createBody(
  world: PhysicalWorld,
  { type, x, y, z, rot = IDENTITY_QUAT }: CreateBodyOptions,
): RigidBody {
  const desc = (type === "fixed" ? RigidBodyDesc.fixed() : RigidBodyDesc.dynamic())
    .setTranslation(x, y, z)
    .setRotation(rot);
  return world.createRigidBody(desc);
}
