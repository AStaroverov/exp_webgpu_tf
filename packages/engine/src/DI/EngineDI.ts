import type { EngineWorld } from "../ECS/createEngineWorld.ts";
import type { StructuralOp } from "../Physics/opChannel.ts";

// World + the frame entry point. A plain module-level mutable object (not a class),
// filled by createEngine.
//
// At Step 3 there is NO main-thread Rapier world — physics runs in the worker. Pose
// reaches main through the shared SAB; structural changes reach the worker through
// postOps (the op-posting seam, wired to the physics Worker in createEngine).
export type EngineApi = {
  width: number;
  height: number;
  world: EngineWorld;
  // Post batched structural ops (spawn/despawn bodies) to the physics worker. Batch
  // per phase boundary; do NOT post per op (plan §4.3). RigidShapes + demo route every
  // structural change through this — there is no main-thread physics world to mutate.
  postOps: (ops: readonly StructuralOp[]) => void;
  tick: (delta: number) => void;
  destroy: () => void;
  setRenderTarget: (canvas: HTMLCanvasElement | null | undefined) => void;
};

export const EngineDI: EngineApi = {} as EngineApi;
