import { World, Vector3, RigidBodyDesc } from "@dimforge/rapier3d-simd";

// Re-export of the Rapier 3D World as the engine's physics-world type, so call
// sites depend on this alias and not the package path directly.
export type PhysicalWorld = World;

// Z-up world (the renderer's ground is the X-Y plane), so gravity points down −Z.
// See createEngineWorld / the §1 coordinate convention in the integration plan.
export function initPhysicalWorld(): PhysicalWorld {
  const gravity = new Vector3(0, 0, -9.81);
  const world = new World(gravity);

  // Match the unknown package's world scale (cm-ish units) and a cheap solver
  // budget. Both are exposed as plain setters on World in rapier3d-simd@0.19.
  world.lengthUnit = 100;
  world.numSolverIterations = 4;

  reserveHandleZero(world);
  return world;
}

// Rapier handle 0 is the empty-memory sentinel (a zero-filled RigidBodyRef row
// reads 0). Create one throwaway fixed body up front so handle 0 is never a real
// entity — mirrors the unknown package's physics bootstrap.
function reserveHandleZero(world: PhysicalWorld): void {
  world.createRigidBody(RigidBodyDesc.fixed());
}
