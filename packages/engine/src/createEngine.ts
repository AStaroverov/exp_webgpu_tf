import { createTransformSystem } from "../../renderer/src/ECS/Systems/TransformSystem.ts";
import { createEngineWorld, getEngineComponents, getEngineSab } from "./ECS/createEngineWorld.ts";
import { createApplyRigidBodyToTransformSystem } from "./ECS/Systems/createApplyRigidBodyToTransformSystem.ts";
import { createPhysicsWorker } from "./Physics/createPhysicsWorker.ts";
import { createRenderTarget } from "./createRenderTarget.ts";
import { EngineDI, type EngineApi } from "./DI/EngineDI.ts";
import { RenderDI } from "./DI/RenderDI.ts";

export type CreateEngineOptions = {
  canvas?: HTMLCanvasElement | null;
  width?: number;
  height?: number;
};

export async function createEngine({
  canvas,
  width = 0,
  height = 0,
}: CreateEngineOptions = {}): Promise<EngineApi> {
  if (typeof globalThis.crossOriginIsolated === "boolean" && !globalThis.crossOriginIsolated) {
    throw new Error(
      "createEngine: crossOriginIsolated === false. The engine requires " +
        "SharedArrayBuffer; set COOP/COEP headers (Cross-Origin-Opener-Policy: " +
        "same-origin, Cross-Origin-Embedder-Policy: require-corp) on the server. " +
        "There is no single-thread fallback.",
    );
  }

  const world = createEngineWorld();
  const physicsWorker = createPhysicsWorker(getEngineSab(world).bundle);

  const execTransformSystem = createTransformSystem(world, getEngineComponents(world).Children);
  const applyRigidBodyToLocalTransform = createApplyRigidBodyToTransformSystem(world);

  function tick(delta: number): void {
    applyRigidBodyToLocalTransform();
    execTransformSystem();
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
    physicsWorker.terminate();
  }

  EngineDI.width = width;
  EngineDI.height = height;
  EngineDI.world = world;
  EngineDI.tick = tick;
  EngineDI.destroy = destroy;
  EngineDI.setRenderTarget = (target) => void setRenderTarget(target);

  if (canvas) {
    await createRenderTarget(world, canvas);
    EngineDI.width = canvas.width;
    EngineDI.height = canvas.height;
  }

  await physicsWorker.ready;

  return EngineDI;
}
