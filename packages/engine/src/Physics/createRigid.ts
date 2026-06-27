import { ColliderDesc, type RigidBody } from "@dimforge/rapier3d-simd";
import type { PhysicalWorld } from "./initPhysicalWorld.ts";

// Attach a box collider (3 half-extents) to a body and return its numeric handle
// (the pid stored in RigidBodyRef).
export function createRigidBox(
  world: PhysicalWorld,
  body: RigidBody,
  hx: number,
  hy: number,
  hz: number,
): number {
  const desc = ColliderDesc.cuboid(hx, hy, hz).setDensity(1);
  world.createCollider(desc, body);
  return body.handle;
}

// Attach a ball collider (radius) to a body and return its numeric handle.
export function createRigidBall(world: PhysicalWorld, body: RigidBody, radius: number): number {
  const desc = ColliderDesc.ball(radius).setDensity(1);
  world.createCollider(desc, body);
  return body.handle;
}
