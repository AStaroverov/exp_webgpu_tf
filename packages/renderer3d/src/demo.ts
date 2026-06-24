// renderer3d 2.5D true-3D-SDF demo.
//
// Minimal runnable ECS scene mirroring the engine wiring (createWorld → initWebGPU
// → frame textures → draw system → resize/transform systems → present), but
// stripped to renderer3d. Spawns one of every ShapeKind with varied baseZ / height
// / yaw / roundness / color, including a box + sphere stacked on a raised platform
// so "положение Z" (baseZ, the bottom) reads distinctly from honest height.
//
// DEPTH CONVENTION — REVERSE-Z (NEAR=1 .. FAR=0): the draw pipeline compares
// depth "greater-equal" against a 0 clear; ResizeSystem.viewProjMatrix and the
// shader's frag_depth both follow it. See sdf.shader.ts / ResizeSystem.ts.

import { initWebGPU } from "./gpu.ts";
import { createWorld, getRenderComponents } from "./ECS/world.ts";
import { createFrameTextures, createFrameTick } from "./WGSL/createFrame.ts";
import { createPresent } from "./WGSL/createPresent.ts";
import { createDrawShapeSystem } from "./ECS/Systems/SDFSystem/createDrawShapeSystem.ts";
import { createTransformSystem } from "./ECS/Systems/TransformSystem.ts";
import {
  cameraAzimuth,
  cameraZoom,
  createResizeSystem,
  setCameraPosition,
} from "./ECS/Systems/ResizeSystem.ts";
import { applyMatrixRotateZ } from "./ECS/Components/Transform.ts";
import {
  createCircle,
  createParallelogram,
  createRectangle,
  createSphere,
  createTrapezoid,
  createTriangle,
} from "./ECS/Entities/Shapes.ts";

// The transform system multiplies parents by their children; this demo has no
// hierarchy, so a stub Children with zero counts is enough.
const stubChildren = {
  entitiesCount: { get: (_eid: number) => 0 },
  entitiesIds: { get: (_eid: number, _i: number) => 0 },
};

async function main() {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const { device, context } = await initWebGPU(canvas);

  const getPixelRatio = () => window.devicePixelRatio;

  const world = createWorld();
  const { LocalTransform } = getRenderComponents(world);

  // --- Scene: at least one of every kind, plus a stacked platform. ---
  // Ground slab (flat-ish wide rectangle at baseZ below 0).
  createRectangle(world, {
    x: 0, y: 0, z: -0.4, width: 52, height: 52, depth: 0.4,
    color: [0.16, 0.18, 0.22, 1],
  });

  // Sphere (true 3D) — height derived from radius.
  createSphere(world, { x: -6, y: 2, z: 0, radius: 2.2, color: [0.9, 0.8, 0.35, 1] });

  // Cylinder (Circle extruded by height).
  createCircle(world, { x: -3, y: -7, z: 0, radius: 2.5, height: 3, color: [0.4, 0.7, 0.9, 1] });

  // Box (Rectangle extruded) — tall tower, slight yaw.
  const tower = createRectangle(world, {
    x: -9, y: -6, z: 0, width: 4, height: 4, depth: 6, color: [0.85, 0.45, 0.3, 1],
  });
  applyMatrixRotateZ(LocalTransform.matrix.getBatch(tower), 0.4);

  // Parallelogram (extruded), skewed.
  createParallelogram(world, {
    x: -9, y: 6, z: 0, width: 3, height: 5, skew: 1.2, depth: 2.5,
    color: [0.7, 0.5, 0.85, 1],
  });

  // Trapezoid (extruded), rounded.
  createTrapezoid(world, {
    x: -3, y: 7, z: 0, topWidth: 5, bottomWidth: 2, height: 4, depth: 2.5, roundness: 0.3,
    color: [0.85, 0.6, 0.3, 1],
  });

  // Triangle (extruded), rotated.
  const tri = createTriangle(world, {
    x: 3, y: 7, z: 0, depth: 2.5,
    point1: [0, 2.2], point2: [-2.2, -2.2], point3: [2.2, -2.2],
    color: [0.55, 0.85, 0.9, 1],
  });
  applyMatrixRotateZ(LocalTransform.matrix.getBatch(tri), 0.6);

  // Platform + a rounded box standing ON it (baseZ = platform top) + sphere on top.
  createRectangle(world, {
    x: 9, y: 7, z: 0, width: 8, height: 8, depth: 1.5, color: [0.3, 0.32, 0.4, 1],
  });
  createRectangle(world, {
    x: 9, y: 7, z: 1.5, width: 2.8, height: 2.8, depth: 3, roundness: 0.4,
    color: [0.95, 0.55, 0.55, 1],
  });
  createSphere(world, { x: 9, y: 7, z: 4.5, radius: 1, color: [0.95, 0.95, 0.95, 1] });

  // --- Systems ---
  const execTransformSystem = createTransformSystem(world, stubChildren);
  const shapeSystem = createDrawShapeSystem({ world, device });
  const present = createPresent(device, context);

  setCameraPosition(0, 0);
  cameraZoom.value = 14; // pixels per world unit; smaller shows more world

  let frame = createFrameTextures(device, canvas);
  let frameW = canvas.width;
  let frameH = canvas.height;
  let frameTick = createFrameTick(
    { ...frame, canvas, device, background: [0.043, 0.051, 0.07, 1], getPixelRatio },
    ({ passEncoder }) => shapeSystem.drawShapes(passEncoder),
  );

  // Standalone resize/camera update, run BEFORE prepare() so the camera uniforms
  // uploaded each frame are current. (createFrameTick has its own internal resize
  // system, but it runs inside the main pass — i.e. after prepare — which would
  // leave the orbiting camera one frame stale. The internal one then no-ops.)
  const resizeSystem = createResizeSystem(canvas, getPixelRatio);

  let last = performance.now();
  function loop(now: number) {
    const delta = Math.min(now - last, 16.6667);
    last = now;

    // Slowly orbit so the volumes read as 3D.
    cameraAzimuth.value += (delta / 1000) * 18; // ~18 deg/sec

    // Update camera + canvas size first, so prepare() uploads current uniforms
    // and the resize check below sees this frame's dimensions.
    resizeSystem();

    // Recreate frame textures + tick if the canvas was resized.
    if (canvas.width !== frameW || canvas.height !== frameH) {
      frame = createFrameTextures(device, canvas);
      frameW = canvas.width;
      frameH = canvas.height;
      frameTick = createFrameTick(
        { ...frame, canvas, device, background: [0.043, 0.051, 0.07, 1], getPixelRatio },
        ({ passEncoder }) => shapeSystem.drawShapes(passEncoder),
      );
    }

    execTransformSystem();
    shapeSystem.prepare();

    const encoder = device.createCommandEncoder();
    const { renderTexture } = frameTick(encoder, delta);
    present(encoder, renderTexture);
    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f88;padding:20px">${(err as Error)?.stack ?? err}</pre>`;
  console.error(err);
});
