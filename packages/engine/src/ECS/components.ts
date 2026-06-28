import type { World } from "bitecs";
import { createRigidBodyRefComponent } from "./Components/RigidBodyRef.ts";
import { createRigidBodyStateComponent } from "./Components/RigidBodyState.ts";

// The engine-specific (physics-bridge) components, spread alongside the renderer
// components into the world context by createEngineWorld. RigidBodyState reaches the
// shared DATA SAB through ctx.sab (read off world.sab), so the SAB is no longer threaded
// through this factory — the world just needs context.sab set before this runs (plan §3).
export function createEngineComponents(world: World) {
  return {
    RigidBodyRef: createRigidBodyRefComponent(world),
    RigidBodyState: createRigidBodyStateComponent(world),
  };
}

export type EngineComponentsLocal = ReturnType<typeof createEngineComponents>;
