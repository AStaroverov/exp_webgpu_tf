import type { World } from "bitecs";
import { createRigidBodyRefComponent } from "./Components/RigidBodyRef.ts";
import { createRigidBodyStateComponent } from "./Components/RigidBodyState.ts";

export function createEngineComponents(world: World) {
  return {
    RigidBodyRef: createRigidBodyRefComponent(world),
    RigidBodyState: createRigidBodyStateComponent(world),
  };
}

export type EngineComponentsLocal = ReturnType<typeof createEngineComponents>;
