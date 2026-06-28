import { hasComponent, query } from "bitecs";
import { initWebGPU } from "../../renderer3d_2/src/gpu.ts";
import { createFrameTextures, createFrameTick } from "../../renderer3d_2/src/WGSL/createFrame.ts";
import { createPresent } from "../../renderer3d_2/src/WGSL/createPresent.ts";
import { createDrawShapeSystem } from "../../renderer3d_2/src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts";
import { createVoxelSystem } from "../../renderer3d_2/src/ECS/Systems/Lighting/createVoxelSystem.ts";
import { createTransformSystem } from "../../renderer3d_2/src/ECS/Systems/TransformSystem.ts";
import { createResizeSystem } from "../../renderer3d_2/src/ECS/Systems/ResizeSystem.ts";
import { SunLight } from "../../renderer3d_2/src/ECS/Systems/SunLight.ts";
import { getMatrixTranslation } from "../../renderer3d_2/src/ECS/Components/Transform.ts";
import type { EngineWorld } from "./ECS/createEngineWorld.ts";
import { getEngineComponents } from "./ECS/createEngineWorld.ts";
import { RenderDI } from "./DI/RenderDI.ts";
import { stubChildren } from "./lib/constants.ts";

// Builds the GPU render target and the per-frame render closure. Mirrors
// renderer3d_2/src/demo.ts's render path function-for-function (the §4 render
// block), including the resize-rebuild of the frame textures + voxel.recreate.
// Fills RenderDI and returns { renderFrame, destroy }.
export async function createRenderTarget(
  world: EngineWorld,
  canvas: HTMLCanvasElement,
): Promise<{ renderFrame: (delta: number) => void; destroy: () => void }> {
  const { device, context } = await initWebGPU(canvas);

  const getPixelRatio = () => window.devicePixelRatio;

  const { LocalTransform, LightEmitter, Shape, Color } = getEngineComponents(world);

  // --- Systems ---
  const execTransformSystem = createTransformSystem(world, stubChildren);
  const shapeSystem = createDrawShapeSystem({ world, device });
  const present = createPresent(device, context);

  let frame = createFrameTextures(device, canvas);
  let frameW = canvas.width;
  let frameH = canvas.height;
  let frameTick = createFrameTick(
    { ...frame, canvas, device, background: [0.043, 0.051, 0.07, 1], getPixelRatio },
    ({ passEncoder }) => shapeSystem.drawShapes(passEncoder),
  );

  // Voxel GI: voxelize the scene, build the radiance pyramid, cone-gather + composite.
  const voxel = createVoxelSystem({
    device,
    canvas,
    sceneInstances: shapeSystem.sceneInstances,
    depthTexture: frame.depthTexture,
    normalTexture: frame.normalTexture,
    albedoTexture: frame.renderTexture,
    emissionTexture: frame.emissionTexture,
  });

  // Standalone resize/camera update, run BEFORE prepare() so the camera uniforms
  // uploaded each frame are current (mirrors demo.ts; createFrameTick has its own
  // internal resize system, which then no-ops).
  const resizeSystem = createResizeSystem(canvas, getPixelRatio);

  // Auto-discover every emitter from the ECS each frame → cone importance-sampling
  // lights (capped at 8 by voxel.setLights). The demo uses only the sun, so this is
  // usually a no-op (count 0 → pure Fibonacci fill cones), but it keeps the emitter
  // door open at zero cost.
  const emitterLightsFlat: number[] = [];
  const emitterColorsFlat: number[] = [];
  function updateLights() {
    emitterLightsFlat.length = 0;
    emitterColorsFlat.length = 0;
    const ents = query(world, [LightEmitter, LocalTransform]);
    let n = 0;
    for (let i = 0; i < ents.length && n < 8; i++) {
      const id = ents[i];
      const t = getMatrixTranslation(LocalTransform.matrix.getBatch(id));
      // Emitter radius from the shape's first value slot (sphere/circle radius); default 0.5.
      const r = hasComponent(world, id, Shape) ? Shape.values.get(id, 0) || 0.5 : 0.5;
      emitterLightsFlat.push(t[0], t[1], t[2], r);
      const c = Color.getArray(id);
      emitterColorsFlat.push(c[0], c[1], c[2], LightEmitter.intensity[id]);
      n++;
    }
    voxel.setLights(emitterLightsFlat, n, emitterColorsFlat);
  }

  function renderFrame(delta: number) {
    // Update camera + canvas size first, so prepare() uploads current uniforms and
    // the resize check below sees this frame's dimensions.
    resizeSystem();

    // Recreate frame textures + tick if the canvas was resized; rebind the voxel
    // G-buffer + recreate its canvas-sized outputs.
    if (canvas.width !== frameW || canvas.height !== frameH) {
      frame = createFrameTextures(device, canvas);
      frameW = canvas.width;
      frameH = canvas.height;
      frameTick = createFrameTick(
        { ...frame, canvas, device, background: [0.043, 0.051, 0.07, 1], getPixelRatio },
        ({ passEncoder }) => shapeSystem.drawShapes(passEncoder),
      );
      voxel.recreate(
        frame.depthTexture,
        frame.normalTexture,
        frame.renderTexture,
        frame.emissionTexture,
      );
    }

    // LocalTransform → GlobalTransform (the matrices the draw pass reads).
    execTransformSystem();
    // Auto-discover emitters (positions are now current) → cone lights.
    updateLights();
    shapeSystem.prepare();

    const encoder = device.createCommandEncoder();
    // Order (must not be reordered): SDF G-buffer draw → (sun depth, only when the
    // directional sun is on) → voxelize → mips → probe → probeBlur → cone → composite → present.
    frameTick(encoder, delta);
    if (SunLight.enabled) {
      voxel.sunDepth(encoder);
    }
    voxel.voxelize(encoder);
    voxel.mips(encoder);
    voxel.probe(encoder);
    voxel.probeBlur(encoder);
    voxel.cone(encoder);
    voxel.composite(encoder);
    present(encoder, voxel.compositeOutputTexture);
    device.queue.submit([encoder.finish()]);
  }

  RenderDI.enabled = true;
  RenderDI.canvas = canvas;
  RenderDI.device = device;
  RenderDI.context = context;
  RenderDI.renderFrame = renderFrame;
  // Expose the VCT system so a host (demo) can build a tuning GUI against voxel.config.
  RenderDI.voxel = voxel;

  function destroy() {
    RenderDI.enabled = false;
    RenderDI.renderFrame = undefined;
    RenderDI.voxel = undefined;
  }
  RenderDI.destroy = destroy;

  return { renderFrame, destroy };
}
