// Shared tiny constants used across the physics↔render seam.

import type { Rotation } from "@dimforge/rapier3d-simd";

// Identity quaternion (no rotation). Rapier's Rotation interface is {x,y,z,w};
// the identity is (0,0,0,1). Plain objects are accepted anywhere a Rotation is
// expected, so this constant doubles as the default `rot` for body factories.
export const IDENTITY_QUAT: Rotation = { x: 0, y: 0, z: 0, w: 1 };
