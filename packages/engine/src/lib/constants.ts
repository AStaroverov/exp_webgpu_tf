// Shared tiny constants used across the physics↔render seam.

import type { Rotation } from "@dimforge/rapier3d-simd";

// Identity quaternion (no rotation). Rapier's Rotation interface is {x,y,z,w};
// the identity is (0,0,0,1). Plain objects are accepted anywhere a Rotation is
// expected, so this constant doubles as the default `rot` for body factories.
export const IDENTITY_QUAT: Rotation = { x: 0, y: 0, z: 0, w: 1 };

// The transform system multiplies parents by their children; the engine demo has
// no hierarchy, so a stub Children with zero counts is enough to satisfy
// createTransformSystem's second query (it iterates zero children per entity).
export const stubChildren = {
  entitiesCount: { get: (_eid: number) => 0 },
  entitiesIds: { get: (_eid: number, _i: number) => 0 },
};
