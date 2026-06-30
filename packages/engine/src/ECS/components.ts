import type { World } from "bitecs";
import { createRigidBodyRefComponent } from "./Components/RigidBodyRef.ts";
import { createRigidBodyStateComponent } from "./Components/RigidBodyState.ts";
import { createChildrenComponent } from "./Components/Children.ts";
import { createVelocityComponent } from "./Components/Velocity.ts";

export function createEngineComponents(world: World) {
  return {
    Children: createChildrenComponent(world),
    RigidBodyRef: createRigidBodyRefComponent(world),
    RigidBodyState: createRigidBodyStateComponent(world),
    Velocity: createVelocityComponent(world),
  };
}
