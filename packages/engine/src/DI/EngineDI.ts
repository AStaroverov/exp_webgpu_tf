import type { EngineWorld } from "../ECS/createEngineWorld.ts";

// World + the frame entry point. A plain module-level mutable object (not a class),
// filled by createEngine.
//
// At Step 3 there is NO main-thread Rapier world — physics runs in the worker. Pose
// reaches main through the shared SAB. Structural changes are PURE ECS: adding
// RigidBodyState (with a body spec) emits the spawn op, removeEntity emits the despawn op
// (both via the OPS ring SAB, inside the component) — so there is no postOps method here.
export type EngineApi = {
  width: number;
  height: number;
  world: EngineWorld;
  tick: (delta: number) => void;
  destroy: () => void;
  setRenderTarget: (canvas: HTMLCanvasElement | null | undefined) => void;
};

export const EngineDI: EngineApi = {} as EngineApi;
