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
import {
  applyMatrixRotateZ,
  getMatrixTranslation,
  setMatrixRotateZ,
  setMatrixTranslate,
} from "./ECS/Components/Transform.ts";
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
  const SCENE = "final" as "emitter" | "showcase" | "final";

  // The configurable emitter (only used by the "emitter" scene). Live-edited from
  // the GUI: position via the transform (re-uploaded every frame), radius via the
  // Shape + Height setters (onSet → re-collected), intensity via LightEmitter.set$.
  const emitterCfg = { x: -6, y: 0, z: 2.5, radius: 2.5, intensity: 2.0 };
  let emitterId = -1;

  // Dynamic objects of the "final" scene, animated every frame (so the change re-voxelizes
  // and the GI updates live). -1 = not in this scene. Each demonstrates ONE animated axis.
  const finalDyn = { animate: true, speed: 1.0 };
  let dynOrbitEmitter = -1; // position — orbits the scene
  let dynRotBox = -1; // angle — spins about Z (moving occluder)
  let dynSizeSphere = -1; // size — radius pulses
  let dynPulseEmitter = -1; // intensity — emission pulses

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

  // --- Final scene: static set + 4 dynamic objects (one per animated axis). ---
  if (SCENE === "final") {
    // Soft warm sun from above: lifts surfaces the emitters don't reach. The GI (indirect
    // bounce + AO) does most of the shaping; the sun is a gentle key.
    SunLight.enabled = true;
    SunLight.angle = 2.4;
    SunLight.elevation = 0.95;
    SunLight.intensity = 0.9;
    SunLight.color = [1.0, 0.93, 0.82];

    // Static set — ground + a small cluster of structures (witnesses for color bleed/AO).
    createRectangle(world, {
      x: 0,
      y: 0,
      z: -0.4,
      width: 60,
      height: 60,
      depth: 0.4,
      color: [0.22, 0.21, 0.18, 1],
    });
    createRectangle(world, {
      x: -10,
      y: 9,
      z: 0,
      width: 10,
      height: 2,
      depth: 7,
      color: [0.7, 0.7, 0.72, 1],
    }); // back wall
    createRectangle(world, {
      x: 9,
      y: 9,
      z: 0,
      width: 3,
      height: 3,
      depth: 5,
      color: [0.75, 0.55, 0.4, 1],
    }); // warm pillar
    createCircle(world, { x: 10, y: -2, z: 0, radius: 1.8, height: 4, color: [0.5, 0.7, 0.6, 1] }); // column
    createSphere(world, { x: 2, y: -9, z: 0, radius: 1.6, color: [0.9, 0.9, 0.92, 1] }); // white witness sphere

    // (1) POSITION — a warm emitter orbiting the scene at mid height.
    dynOrbitEmitter = createSphere(world, {
      x: 9,
      y: 0,
      z: 3,
      radius: 3,
      color: [1.0, 0.55, 0.2, 1],
    });
    LightEmitter.addComponent(world, dynOrbitEmitter, 30.0);
    // (2) ANGLE — a tall slab rotating about Z: a moving occluder → shifting AO/bounce.
    dynRotBox = createRectangle(world, {
      x: 0,
      y: 3,
      z: 0,
      width: 5,
      height: 1.4,
      depth: 6,
      color: [0.8, 0.45, 0.5, 1],
    });
    // (3) SIZE — a solid sphere whose radius pulses (re-voxelized each frame).
    dynSizeSphere = createSphere(world, {
      x: -9,
      y: -6,
      z: 0,
      radius: 2.0,
      color: [0.45, 0.6, 0.85, 1],
    });
    // (4) INTENSITY — a cool emitter pulsing its emission.
    dynPulseEmitter = createSphere(world, {
      x: -9,
      y: 2,
      z: 2.2,
      radius: 0.9,
      color: [0.35, 0.6, 1.0, 1],
    });
    LightEmitter.addComponent(world, dynPulseEmitter, 2.5);
  }

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
    depthTexture: frame.depthTexture,
    normalTexture: frame.normalTexture,
    albedoTexture: frame.renderTexture,
    emissionTexture: frame.emissionTexture,
  });

  // Present-source selector: which texture reaches the screen.
  //   "voxel"    — the voxel debug raymarch, lit-albedo Lambert (voxel.outputTexture).
  //   "radiance" — the voxel debug raymarch showing stored direct-sun radiance
  //                (voxel.outputTexture, debug mode 1).
  //   "lod"      — the voxel debug raymarch sampling the radiance mip pyramid at a chosen
  //                LOD (voxel.outputTexture, debug mode 2); LOD 0 sharp, higher = blurred.
  //   "cone"     — VCT cone GI: 6-cone diffuse hemisphere gather through the radiance
  //                pyramid (voxel.coneOutputTexture); indirect only (no albedo/direct yet).
  //   "final"    — the composited VCT image (voxel.compositeOutputTexture): the real lit
  //                picture = albedo·(ambient·AO + directSun + indirect) + selfEmission.
  //   "raw"      — the unlit albedo G-buffer (frame.renderTexture).
  // Keys: 1 = voxel, 5 = radiance, 6 = lod, 7 = cone, 8 = final, 2 = raw (also a GUI dropdown).
  const view = {
    presentSource: "final" as "voxel" | "raw" | "radiance" | "lod" | "cone" | "final",
  };
  window.addEventListener("keydown", (e) => {
    if (e.key === "1") view.presentSource = "voxel";
    else if (e.key === "2") view.presentSource = "raw";
    else if (e.key === "5") view.presentSource = "radiance";
    else if (e.key === "6") view.presentSource = "lod";
    else if (e.key === "7") view.presentSource = "cone";
    else if (e.key === "8") view.presentSource = "final";
  });

  const gui = new GUI({ title: "Voxel" });
  gui
    .add(view, "presentSource", ["raw", "voxel", "radiance", "lod", "cone", "final"])
    .name("present")
    .listen();

  // Debug LOD for the "lod" present source: which mip of the radiance pyramid to sample.
  const lodCfg = { lod: 0 };
  gui.add(lodCfg, "lod", 0, voxel.mipCount - 1, 1).name("debug LOD");

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
  gui.add(SunLight, "elevation", 0, Math.PI / 2, 0.01).name("sun elevation");
  gui.add(SunLight, "intensity", 0, 5, 0.05).name("sun intensity");
  gui.addColor(SunLight, "color", 1).name("sun color"); // rgbScale=1 → array is 0..1 floats

  // Cone GI: the N-cone diffuse hemisphere trace. Shadows emerge from this single gather —
  // the cone count is the angular resolution: more cones + narrower aperture = sharper
  // shadows (but too-narrow + too-few cones bands). Read live each frame by cone().
  const coneFolder = gui.addFolder("Cone GI");
  coneFolder.add(voxel.coneParams, "coneCount", [6, 8, 16, 24, 32, 48]).name("cones (shadow sharpness)");
  coneFolder.add(voxel.coneParams, "aperture", 0.1, 1.5, 0.01).name("aperture (lower=sharper)");
  coneFolder.add(voxel.coneParams, "maxDist", 1, 64, 0.5).name("cone reach");
  coneFolder.add(voxel.coneParams, "normalBias", 0, 2, 0.01).name("normal bias");
  coneFolder.add(voxel.coneParams, "giStrength", 0, 4, 0.05).name("GI strength");

  // Composite (Layer 4): the final lit image. Sun controls above feed it via SunLight;
  // cone giStrength bakes into the indirect term. Only the ambient floor lives here.
  const compositeFolder = gui.addFolder("Composite");
  compositeFolder.add(voxel.compositeParams, "ambient", 0, 0.5, 0.01).name("composite ambient");

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

  // Final-scene animation controls.
  if (dynOrbitEmitter >= 0) {
    const f = gui.addFolder("Final scene");
    f.add(finalDyn, "animate").name("animate");
    f.add(finalDyn, "speed", 0, 3, 0.05).name("speed");
  }

  // Emitters the cone pass importance-samples (aimed sharp soft-shadow cones). Only the
  // "final" scene tracks any; .radius feeds the penumbra width + the sphere-center Z offset
  // (SDF center.z = baseZ + radius). Filtered to live ids (>= 0).
  const finalLights = [
    { id: dynOrbitEmitter, radius: 0.8 },
    { id: dynPulseEmitter, radius: 0.9 },
  ].filter((l) => l.id >= 0);
  // Flat scratch (x,y,z,radius per light) reused every frame for voxel.setLights.
  const lightsFlat: number[] = [];

  // Recompute the importance-sampled light centers from the current transforms and push them
  // to the cone pass. center = LocalTransform translation + (0,0,radius). Empty when the scene
  // has no tracked emitters → count 0 → pure fill cones (old behavior).
  function updateLights() {
    lightsFlat.length = 0;
    for (let i = 0; i < finalLights.length; i++) {
      const { id, radius } = finalLights[i];
      const t = getMatrixTranslation(LocalTransform.matrix.getBatch(id));
      lightsFlat.push(t[0], t[1], t[2] + radius, radius);
    }
    voxel.setLights(lightsFlat, finalLights.length);
  }

  // Animate the "final" scene: position (orbit), angle (spin), size (pulse radius),
  // intensity (pulse emission). Mutates ECS state BEFORE prepare()/voxelize so the change
  // re-voxelizes this frame and the GI follows it live.
  function animateFinal(now: number) {
    if (dynOrbitEmitter < 0 || !finalDyn.animate) return;
    const t = now * 0.001 * finalDyn.speed;
    // (1) position — orbit on a circle of radius 9 at height 3.
    setMatrixTranslate(
      LocalTransform.matrix.getBatch(dynOrbitEmitter),
      Math.cos(t * 0.6) * 9,
      Math.sin(t * 0.6) * 9,
      3,
    );
    // (2) angle — spin the slab about Z (its translation persists from creation).
    setMatrixRotateZ(LocalTransform.matrix.getBatch(dynRotBox), t * 0.8);
    // (3) size — radius 1..3, re-voxelized via the Shape + Height setters (Height = 2·r keeps
    // the bottom on the ground, since the SDF center.z = baseZ + Height/2).
    const r = 2.0 + 1.0 * Math.sin(t * 1.2);
    Shape.setSphere$(dynSizeSphere, r);
    Height.set$(dynSizeSphere, r * 2);
    // (4) intensity — emission 0.2..3.8 (kept positive; negative would mean "directional").
    LightEmitter.set$(dynPulseEmitter, 2.0 + 1.8 * Math.sin(t * 1.6), 0);
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
  function loop(now: number) {
    const delta = Math.min(now - last, 16.6667);
    last = now;

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
      // Voxel: rebind the new G-buffer + recreate its canvas-sized outputs (debug + cone +
      // GI accumulation).
      voxel.recreate(
        frame.depthTexture,
        frame.normalTexture,
        frame.renderTexture,
        frame.emissionTexture,
      );
    }

    // Drive the dynamic "final"-scene objects, then push transforms → instance buffers.
    animateFinal(now);
    // Refresh the importance-sampled emitter centers for the cone pass (positions are now
    // current). Empty when no emitters are tracked → count 0 → pure fill cones.
    updateLights();
    execTransformSystem();
    shapeSystem.prepare();

    const encoder = device.createCommandEncoder();
    // Main SDF draw pass -> renderTexture (raw albedo G-buffer). Its per-fragment
    // sphere-trace (up to 96 steps) is fill-bound: cost scales with on-screen coverage,
    // so it spikes on zoom-in. "voxel" doesn't read the G-buffer (only the voxel 3D
    // textures + scene buffers, which prepare() already uploaded) — it uses the voxel
    // DDA instead. "raw", "cone" and "final" all NEED the SDF pass: raw presents it; cone +
    // final read its depth + normal G-buffer to reconstruct P + N (final also reads albedo).
    if (
      view.presentSource === "raw" ||
      view.presentSource === "cone" ||
      view.presentSource === "final"
    ) {
      frameTick(encoder, delta);
    }
    // Voxelize the SDF scene into the 3D textures, then run the selected voxel pass.
    voxel.voxelize(encoder);
    if (view.presentSource === "voxel") voxel.debug(encoder, 0);
    else if (view.presentSource === "radiance") voxel.debug(encoder, 1);
    else if (view.presentSource === "lod") {
      // Build the radiance mip pyramid (must follow voxelize), then sample the chosen LOD.
      voxel.mips(encoder);
      voxel.debug(encoder, 2, lodCfg.lod);
    } else if (view.presentSource === "cone") {
      // Build the radiance pyramid (must follow voxelize), then trace the diffuse cones.
      voxel.mips(encoder);
      voxel.cone(encoder);
    } else if (view.presentSource === "final") {
      // Pyramid → cone gather → composite into the final lit image (composite reads cone).
      voxel.mips(encoder);
      voxel.cone(encoder);
      voxel.composite(encoder);
    }
    // Present the chosen source.
    const presented =
      view.presentSource === "voxel" ||
      view.presentSource === "radiance" ||
      view.presentSource === "lod"
        ? voxel.outputTexture
        : view.presentSource === "cone"
          ? voxel.coneOutputTexture
          : view.presentSource === "final"
            ? voxel.compositeOutputTexture
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
