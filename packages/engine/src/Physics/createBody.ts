import { RigidBodyDesc, type RigidBody, type Rotation } from "@dimforge/rapier3d-simd";
import type { PhysicalWorld } from "./initPhysicalWorld.ts";
import { IDENTITY_QUAT } from "../lib/constants.ts";

export type BodyType = "dynamic" | "fixed";

export type CreateBodyOptions = {
  type: BodyType;
  x: number;
  y: number;
  z: number;
  rot?: Rotation;
};

export function createBody(
  world: PhysicalWorld,
  { type, x, y, z, rot = IDENTITY_QUAT }: CreateBodyOptions,
): RigidBody {
  const desc = (type === "fixed" ? RigidBodyDesc.fixed() : RigidBodyDesc.dynamic())
    .setTranslation(x, y, z)
    .setRotation(rot);
  return world.createRigidBody(desc);
}
