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

import GUI from "lil-gui";
import { initWebGPU } from "./gpu.ts";
import { createWorld, getRenderComponents } from "./ECS/world.ts";
import { createFrameTextures, createFrameTick } from "./WGSL/createFrame.ts";
import { createPresent } from "./WGSL/createPresent.ts";
import { createDrawShapeSystem } from "./ECS/Systems/SDFSystem/createDrawShapeSystem.ts";
import { createVoxelSystem } from "./ECS/Systems/Lighting/createVoxelSystem.ts";
import { SunLight } from "./ECS/Systems/SunLight.ts";
import { createTransformSystem } from "./ECS/Systems/TransformSystem.ts";
import {
  cameraAzimuth,
  cameraElevation,
  cameraZoom,
  createResizeSystem,
  setCameraElevation,
  setCameraPosition,
} from "./ECS/Systems/ResizeSystem.ts";
import { computeSmokeTest } from "./WGSL/computeSmokeTest.ts";
import { applyMatrixRotateZ, setMatrixTranslate } from "./ECS/Components/Transform.ts";
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

  // Stage-0 compute-infrastructure check (logs [compute-smoke] PASS/FAIL once).
  void computeSmokeTest(device);

  const getPixelRatio = () => window.devicePixelRatio;

  const world = createWorld();
  const { LocalTransform, LightEmitter, Shape, Height } = getRenderComponents(world);

  // Scene selection:
  //   "emitter"  — ground + box occluder + a GUI-movable/resizable emitter sphere.
  //   "simple"   — fixed minimal scene (first-bug diagnosis).
  //   "showcase" — one of every shape kind + several lights.
  const SCENE = "showcase" as "emitter" | "simple" | "showcase";

  // The configurable emitter (only used by the "emitter" scene). Live-edited from
  // the GUI: position via the transform (re-uploaded every frame), radius via the
  // Shape + Height setters (onSet → re-collected), intensity via LightEmitter.set$.
  const emitterCfg = { x: -6, y: 0, z: 2.5, radius: 2.5, intensity: 2.0 };
  let emitterId = -1;

  if (SCENE === "emitter") {
    SunLight.enabled = false; // isolate the single emitter
    createRectangle(world, {
      x: 0,
      y: 0,
      z: -0.5,
      width: 40,
      height: 40,
      depth: 0.5,
      color: [0.5, 0.5, 0.5, 1],
    });
    createRectangle(world, {
      x: 4,
      y: 0,
      z: 0,
      width: 3,
      height: 3,
      depth: 4,
      color: [0.8, 0.8, 0.8, 1],
    });
    emitterId = createSphere(world, {
      x: emitterCfg.x,
      y: emitterCfg.y,
      z: emitterCfg.z,
      radius: emitterCfg.radius,
      color: [1, 1, 1, 1],
    });
    LightEmitter.addComponent(world, emitterId, emitterCfg.intensity);
  }

  if (SCENE === "simple") {
    SunLight.enabled = false; // isolate the single emitter
    // Ground: 40x40 slab, top surface at z = 0.
    createRectangle(world, {
      x: 0,
      y: 0,
      z: -0.5,
      width: 40,
      height: 40,
      depth: 0.5,
      color: [0.5, 0.5, 0.5, 1],
    });
    // One box occluder at +X (world (4,0), 3x3 footprint, 4 tall).
    createRectangle(world, {
      x: 4,
      y: 0,
      z: 0,
      width: 3,
      height: 3,
      depth: 4,
      color: [0.8, 0.8, 0.8, 1],
    });
    // One LARGE white emissive sphere at -X (world (-6,0,2.5), radius 2.5).
    const lamp = createSphere(world, { x: -6, y: 0, z: 2.5, radius: 2.5, color: [1, 1, 1, 1] });
    LightEmitter.addComponent(world, lamp, 2.0);
  }

  // --- Scene: at least one of every kind, plus a stacked platform. ---
  // Ground slab (flat-ish wide rectangle at baseZ below 0).
  if (SCENE === "showcase") {
    createRectangle(world, {
      x: 0,
      y: 0,
      z: -0.4,
      width: 52,
      height: 52,
      depth: 0.4,
      color: [0.16, 0.18, 0.22, 1],
    });

    // Sphere (true 3D) — height derived from radius.
    createSphere(world, { x: -6, y: 2, z: 0, radius: 2.2, color: [0.9, 0.8, 0.35, 1] });

    // Cylinder (Circle extruded by height).
    createCircle(world, { x: -3, y: -7, z: 0, radius: 2.5, height: 3, color: [0.4, 0.7, 0.9, 1] });

    // Box (Rectangle extruded) — tall tower, slight yaw.
    const tower = createRectangle(world, {
      x: -9,
      y: -6,
      z: 0,
      width: 4,
      height: 4,
      depth: 6,
      color: [0.85, 0.45, 0.3, 1],
    });
    applyMatrixRotateZ(LocalTransform.matrix.getBatch(tower), 0.4);

    // Parallelogram (extruded), skewed.
    createParallelogram(world, {
      x: -9,
      y: 6,
      z: 0,
      width: 3,
      height: 5,
      skew: 1.2,
      depth: 2.5,
      color: [0.7, 0.5, 0.85, 1],
    });

    // Trapezoid (extruded), rounded.
    createTrapezoid(world, {
      x: -3,
      y: 7,
      z: 0,
      topWidth: 5,
      bottomWidth: 2,
      height: 4,
      depth: 2.5,
      roundness: 0.3,
      color: [0.85, 0.6, 0.3, 1],
    });

    // Triangle (extruded), rotated.
    const tri = createTriangle(world, {
      x: 3,
      y: 7,
      z: 0,
      depth: 2.5,
      point1: [0, 2.2],
      point2: [-2.2, -2.2],
      point3: [2.2, -2.2],
      color: [0.55, 0.85, 0.9, 1],
    });
    applyMatrixRotateZ(LocalTransform.matrix.getBatch(tri), 0.6);

    // Platform + a rounded box standing ON it (baseZ = platform top) + sphere on top.
    createRectangle(world, {
      x: 9,
      y: 7,
      z: 0,
      width: 8,
      height: 8,
      depth: 1.5,
      color: [0.3, 0.32, 0.4, 1],
    });
    createRectangle(world, {
      x: 9,
      y: 7,
      z: 1.5,
      width: 2.8,
      height: 2.8,
      depth: 3,
      roundness: 0.4,
      color: [0.95, 0.55, 0.55, 1],
    });
    createSphere(world, { x: 9, y: 7, z: 4.5, radius: 1, color: [0.95, 0.95, 0.95, 1] });

    // --- Light emitters (GI sources). ---
    // intensity > 0 = omni; intensity < 0 = directional (facing = world +X).
    // Warm omni near the tower.
    const warmLamp = createSphere(world, {
      x: -9,
      y: -2,
      z: 1.5,
      radius: 0.7,
      color: [1.0, 0.6, 0.2, 1],
    });
    LightEmitter.addComponent(world, warmLamp, 3.0);

    // Cool omni on the far side.
    const coolLamp = createSphere(world, {
      x: 9,
      y: -6,
      z: 1.2,
      radius: 0.6,
      color: [0.3, 0.6, 1.0, 1],
    });
    LightEmitter.addComponent(world, coolLamp, 2.5);

    // Directional beam (negative intensity): a rectangle facing world +X.
    const beam = createRectangle(world, {
      x: 0,
      y: -2,
      z: 0.6,
      width: 1.2,
      height: 0.6,
      depth: 1,
      color: [0.9, 0.95, 0.7, 1],
    });
    LightEmitter.addComponent(world, beam, -2.5);
  } // end showcase

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

  // Phase 1: voxelize the SDF scene into 3D albedo/emission textures, then raymarch
  // them (DDA) into voxel.outputTexture for debug inspection.
  const voxel = createVoxelSystem({
    device,
    canvas,
    sceneInstances: shapeSystem.sceneInstances,
  });

  // Present-source selector: which texture reaches the screen.
  //   "voxel" — the voxel debug raymarch (voxel.outputTexture).
  //   "gi"    — brute-force voxel GI reference (voxel.giOutputTexture).
  //   "raw"   — the unlit albedo G-buffer (frame.renderTexture).
  // Keys: 1 = voxel, 2 = raw, 3 = gi (also a GUI dropdown below).
  // Default to "voxel" (cheap) so the page never opens straight into a heavy GI pass.
  const view = { presentSource: "voxel" as "voxel" | "raw" | "gi" };
  window.addEventListener("keydown", (e) => {
    if (e.key === "1") view.presentSource = "voxel";
    else if (e.key === "2") view.presentSource = "raw";
    else if (e.key === "3") view.presentSource = "gi";
  });

  const gui = new GUI({ title: "Voxel" });
  gui.add(view, "presentSource", ["voxel", "gi", "raw"]).name("present (1/3/2)").listen();

  // Graininess: voxel size in world units. Smaller = finer = more voxels. Rebuilds the
  // 3D textures on release (.onFinishChange, so it rebuilds once when the slider settles).
  // The displayed dims controller reflects the resulting per-axis voxel counts.
  const voxCfg = { cellSize: voxel.cellSize };
  const dimsLabel = { dims: `${voxel.dims.x}×${voxel.dims.y}×${voxel.dims.z}` };
  const dimsCtl = gui.add(dimsLabel, "dims").name("voxel dims").disable();
  gui
    .add(voxCfg, "cellSize", 0.25, 2, 0.05)
    .name("voxel size (graininess)")
    .onFinishChange((cs: number) => {
      voxel.setCellSize(cs);
      dimsLabel.dims = `${voxel.dims.x}×${voxel.dims.y}×${voxel.dims.z}`;
      dimsCtl.updateDisplay();
    });

  gui.add(voxel.params, "ambient", 0, 1, 0.01).name("voxel ambient");
  // Sun toggle is read live by the draw pass; keep it exposed for the raw view.
  gui.add(SunLight, "enabled").name("sun enabled");
  gui.add(SunLight, "angle", 0, Math.PI * 2, 0.01).name("sun angle");

  // GI (Stage 2.1a brute-force reference). All read live each frame by voxel.gi().
  const giFolder = gui.addFolder("GI (reference)");
  // Resolution divisor: GI renders at 1/scale then upscales. Higher = much cheaper.
  // Rebuilds the GI target on release. THE primary knob if the GI pass is too heavy.
  const giScaleCfg = { scale: voxel.giScale };
  giFolder
    .add(giScaleCfg, "scale", [1, 2, 4, 8])
    .name("resolution 1/N")
    .onChange((s: number) => voxel.setGiScale(s));
  giFolder.add(voxel.giParams, "numRays", 1, 256, 1).name("rays/pixel");
  giFolder.add(voxel.giParams, "maxDist", 1, 64, 0.5).name("ray reach");
  giFolder.add(voxel.giParams, "giStrength", 0, 4, 0.05).name("GI strength");
  giFolder.add(voxel.giParams, "ambient", 0, 1, 0.01).name("ambient");
  giFolder.add(voxel.giParams, "skyIntensity", 0, 2, 0.01).name("sky on miss");
  giFolder.add(voxel.giParams, "normalBias", 0, 2, 0.01).name("normal bias");
  // Temporal denoise: lower = smoother (averages more frames), 1 = off. Auto-resets the
  // frame the camera moves (no reprojection yet → history would smear).
  giFolder.add(voxel.giParams, "accumAlpha", 0.02, 1, 0.01).name("temporal α");

  // Live emitter controls (only the "emitter" scene). Position writes the transform
  // (re-uploaded every frame); radius drives BOTH the sphere SDF (Shape.setSphere$)
  // and its vertical extent (Height.set$ = 2*radius); intensity via LightEmitter.set$.
  if (emitterId >= 0) {
    const moveEmitter = () =>
      setMatrixTranslate(
        LocalTransform.matrix.getBatch(emitterId),
        emitterCfg.x,
        emitterCfg.y,
        emitterCfg.z,
      );
    const resizeEmitter = () => {
      Shape.setSphere$(emitterId, emitterCfg.radius);
      Height.set$(emitterId, emitterCfg.radius * 2);
    };
    const em = gui.addFolder("Emitter");
    em.add(emitterCfg, "x", -20, 20, 0.1).onChange(moveEmitter);
    em.add(emitterCfg, "y", -20, 20, 0.1).onChange(moveEmitter);
    em.add(emitterCfg, "z", 0, 12, 0.1).onChange(moveEmitter);
    em.add(emitterCfg, "radius", 0.2, 8, 0.1).onChange(resizeEmitter);
    em.add(emitterCfg, "intensity", 0, 8, 0.1).onChange(() =>
      LightEmitter.set$(emitterId, emitterCfg.intensity, emitterCfg.radius),
    );
  }

  // Standalone resize/camera update, run BEFORE prepare() so the camera uniforms
  // uploaded each frame are current. (createFrameTick has its own internal resize
  // system, but it runs inside the main pass — i.e. after prepare — which would
  // leave the orbiting camera one frame stale. The internal one then no-ops.)
  const resizeSystem = createResizeSystem(canvas, getPixelRatio);

  // --- Mouse orbit: horizontal drag = azimuth, vertical drag = elevation, wheel = zoom.
  // Pointer capture keeps the drag alive when the cursor leaves the canvas.
  let dragging = false;
  canvas.style.cursor = "grab";
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    canvas.style.cursor = "grabbing";
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", (e) => {
    dragging = false;
    canvas.style.cursor = "grab";
    canvas.releasePointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    cameraAzimuth.value += e.movementX * 0.4; // ~0.4 deg per pixel
    setCameraElevation(cameraElevation.value - e.movementY * 0.3); // clamped to (1, 89.9)
  });
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      cameraZoom.value = Math.max(
        4,
        Math.min(60, cameraZoom.value * (1 - Math.sign(e.deltaY) * 0.1)),
      );
    },
    { passive: false },
  );

  let last = performance.now();
  let frameIndex = 0;
  function loop(now: number) {
    const delta = Math.min(now - last, 16.6667);
    last = now;
    frameIndex++;

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
      // Voxel: recreate its canvas-sized outputs (debug + GI accumulation).
      voxel.recreate();
    }

    execTransformSystem();
    shapeSystem.prepare();

    const encoder = device.createCommandEncoder();
    // Main SDF draw pass -> renderTexture (raw albedo G-buffer). Its per-fragment
    // sphere-trace (up to 96 steps) is fill-bound: cost scales with on-screen coverage,
    // so it spikes on zoom-in. "voxel"/"gi" don't read the G-buffer (only the voxel 3D
    // textures + scene buffers, which prepare() already uploaded) — they use the voxel
    // DDA instead — so the SDF pass runs ONLY for the raw view.
    if (view.presentSource === "raw") frameTick(encoder, delta);
    // Voxelize the SDF scene into the 3D textures, then run the selected voxel pass.
    voxel.voxelize(encoder);
    if (view.presentSource === "voxel") voxel.debug(encoder);
    else if (view.presentSource === "gi") voxel.gi(encoder, frameIndex);
    // Present the chosen source.
    const presented =
      view.presentSource === "voxel"
        ? voxel.outputTexture
        : view.presentSource === "gi"
          ? voxel.giOutputTexture
          : frame.renderTexture;
    present(encoder, presented);
    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f88;padding:20px">${(err as Error)?.stack ?? err}</pre>`;
  console.error(err);
});
