import { createTransformSystem } from "../../renderer3d_2/src/ECS/Systems/TransformSystem.ts";
import { createEngineWorld } from "./ECS/createEngineWorld.ts";
import { initPhysicalWorld } from "./Physics/initPhysicalWorld.ts";
import { createRigidBodyStateSystem } from "./ECS/Systems/createRigidBodyStateSystem.ts";
import { createApplyRigidBodyToTransformSystem } from "./ECS/Systems/createApplyRigidBodyToTransformSystem.ts";
import { createRenderTarget } from "./createRenderTarget.ts";
import { EngineDI, type EngineApi } from "./DI/EngineDI.ts";
import { RenderDI } from "./DI/RenderDI.ts";
import { stubChildren } from "./lib/constants.ts";

export type CreateEngineOptions = {
  canvas?: HTMLCanvasElement | null;
  width?: number;
  height?: number;
};

// Assembles the engine: bitecs world (render + physics components), Rapier world,
// the two physics→render sync systems, and the deterministic tick(). When a canvas
// is supplied it wires the render target; otherwise the world runs headless.
export async function createEngine({
  canvas,
  width = 0,
  height = 0,
}: CreateEngineOptions = {}): Promise<EngineApi> {
  const world = createEngineWorld();
  const physicalWorld = initPhysicalWorld();

  // Hierarchy compose for any static/child offsets (cheap, harmless for flat scenes).
  const execTransformSystem = createTransformSystem(world, stubChildren);
  const syncRigidBodyState = createRigidBodyStateSystem(world, physicalWorld);
  const applyRigidBodyToLocalTransform = createApplyRigidBodyToTransformSystem(world);

  function physicalFrame(_delta: number): void {
    execTransformSystem(); // local→global compose (runs first, mirrors unknown's order)
    physicalWorld.step(); // Rapier integrates (gravity)
    syncRigidBodyState(); // Rapier body → RigidBodyState (READ)
    applyRigidBodyToLocalTransform(); // RigidBodyState → LocalTransform.matrix (mat4)
  }

  function tick(delta: number): void {
    physicalFrame(delta);
    RenderDI.renderFrame?.(delta);
  }

  async function setRenderTarget(target: HTMLCanvasElement | null | undefined): Promise<void> {
    RenderDI.destroy?.();
    if (target) {
      await createRenderTarget(world, target);
      EngineDI.width = target.width;
      EngineDI.height = target.height;
    }
  }

  function destroy(): void {
    RenderDI.destroy?.();
  }

  EngineDI.width = width;
  EngineDI.height = height;
  EngineDI.world = world;
  EngineDI.physicalWorld = physicalWorld;
  EngineDI.tick = tick;
  EngineDI.destroy = destroy;
  // EngineApi types setRenderTarget as sync-returning; the async render setup is
  // awaited internally here (and below) so callers can also await it directly.
  EngineDI.setRenderTarget = (target) => void setRenderTarget(target);

  if (canvas) {
    await createRenderTarget(world, canvas);
    EngineDI.width = canvas.width;
    EngineDI.height = canvas.height;
  }

  return EngineDI;
}
