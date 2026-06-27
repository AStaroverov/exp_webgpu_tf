import type { World } from "bitecs";
import { createRigidBodyRefComponent } from "./Components/RigidBodyRef.ts";
import { createRigidBodyStateComponent } from "./Components/RigidBodyState.ts";

// The engine-specific (physics-bridge) components, spread alongside the renderer
// components into the world context by createEngineWorld.
export function createEngineComponents(world: World) {
  return {
    RigidBodyRef: createRigidBodyRefComponent(world),
    RigidBodyState: createRigidBodyStateComponent(world),
  };
}

export type EngineComponentsLocal = ReturnType<typeof createEngineComponents>;
