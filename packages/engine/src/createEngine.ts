import { createTransformSystem } from "../../renderer/src/ECS/Systems/TransformSystem.ts";
import { createEngineWorld, getEngineSab } from "./ECS/createEngineWorld.ts";
import { createApplyRigidBodyToTransformSystem } from "./ECS/Systems/createApplyRigidBodyToTransformSystem.ts";
import { createPhysicsWorker } from "./Physics/createPhysicsWorker.ts";
import { createRenderTarget } from "./createRenderTarget.ts";
import { EngineDI, type EngineApi } from "./DI/EngineDI.ts";
import { RenderDI } from "./DI/RenderDI.ts";
import { stubChildren } from "./lib/constants.ts";

export type CreateEngineOptions = {
  canvas?: HTMLCanvasElement | null;
  width?: number;
  height?: number;
};

// Assembles the engine: the MAIN bitecs world (render + bridge components) which
// allocates the shared SAB, the PHYSICS WORKER (Rapier runs off-thread), and the
// per-frame work that reads the worker's published pose bank into the renderer. When a
// canvas is supplied it wires the render target; otherwise the world runs headless.
//
// Physics no longer runs on main (plan §6): the worker steps Rapier + publishes pose
// into the shared SAB; main just reads the latest bank each rAF and renders. Structural
// changes (spawn/despawn) reach the worker through EngineDI.postOps.
export async function createEngine({
  canvas,
  width = 0,
  height = 0,
}: CreateEngineOptions = {}): Promise<EngineApi> {
  // Fail loud if the browser is not cross-origin isolated — SharedArrayBuffer is
  // then unavailable and there is NO single-thread fallback (plan §2/§6.4). In
  // node crossOriginIsolated is undefined (SAB always available) so this is skipped.
  if (typeof globalThis.crossOriginIsolated === "boolean" && !globalThis.crossOriginIsolated) {
    throw new Error(
      "createEngine: crossOriginIsolated === false. The engine requires " +
        "SharedArrayBuffer; set COOP/COEP headers (Cross-Origin-Opener-Policy: " +
        "same-origin, Cross-Origin-Embedder-Policy: require-corp) on the server. " +
        "There is no single-thread fallback.",
    );
  }

  const world = createEngineWorld();
  const sab = getEngineSab(world);

  // Spawn the physics worker and hand it the SAB bundle (same shared bytes main just
  // allocated). The worker self-clocks Rapier from here; main never steps physics.
  const physicsWorker = createPhysicsWorker(sab.bundle);

  // Hierarchy compose for any static/child offsets (cheap, harmless for flat scenes).
  const execTransformSystem = createTransformSystem(world, stubChildren);
  const applyRigidBodyToLocalTransform = createApplyRigidBodyToTransformSystem(world);

  function tick(delta: number): void {
    // Read the worker's latest published pose bank → LocalTransform (the RigidBodyState
    // read accessor resolves sab.readBank() per call; main does NOT publish). Snap to
    // the latest bank — no interpolation (plan §5).
    applyRigidBodyToLocalTransform();
    execTransformSystem(); // local→global compose
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
