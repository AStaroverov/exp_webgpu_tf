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
import { createFrameTextures, createFrameTick, createRCTextures } from "./WGSL/createFrame.ts";
import { createPresent } from "./WGSL/createPresent.ts";
import { createDrawShapeSystem } from "./ECS/Systems/SDFSystem/createDrawShapeSystem.ts";
import {
  createWorldRadianceCascadesSystem,
  DEFAULT_WORLD_RC_PARAMS,
} from "./ECS/Systems/Lighting/createWorldRadianceCascadesSystem.ts";
import { createSurfelSystem } from "./ECS/Systems/Lighting/createSurfelSystem.ts";
import { SunLight } from "./ECS/Systems/SunLight.ts";
import { createTransformSystem } from "./ECS/Systems/TransformSystem.ts";
import {
  cameraAzimuth,
  cameraElevation,
  cameraPosition,
  cameraZoom,
  createResizeSystem,
  setCameraElevation,
  setCameraPosition,
} from "./ECS/Systems/ResizeSystem.ts";
import { createWorldRCDiagnostics } from "./diagnostics.ts";
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

    // --- Light emitters (Radiance Cascades sources). ---
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

  // World-space RC: gather sphere-traces the SDF scene into the cascade probe
  // atlases, merge top-down, then composite over the G-buffer -> worldLitTexture
  // (the presented texture). Toggled on/off in the GUI; when off, the raw unlit
  // scene (renderTexture) is presented.
  const rcTextures = createRCTextures(device, canvas, {
    gridX: DEFAULT_WORLD_RC_PARAMS.gridX,
    gridY: DEFAULT_WORLD_RC_PARAMS.gridY,
    gridZ: DEFAULT_WORLD_RC_PARAMS.gridZ,
    dir0W: DEFAULT_WORLD_RC_PARAMS.dir0W,
  });
  const worldRc = createWorldRadianceCascadesSystem({
    device,
    canvas,
    frameTextures: rcTextures,
    sceneTexture: frame.renderTexture,
    depthTexture: frame.depthTexture,
    normalTexture: frame.normalTexture,
    sceneInstances: shapeSystem.sceneInstances,
  });

  // Surfel Radiance Cascades — Stage A: spawn surfels on visible G-buffer surfaces
  // (a compute pass) and debug-draw them as billboard dots over the presented frame.
  const surfel = createSurfelSystem({
    device,
    depthTexture: frame.depthTexture,
    normalTexture: frame.normalTexture,
  });

  // Enable/disable the world RC. When off, the raw unlit scene is presented.
  // (Key R also toggles it.)
  const view = { enableWorldRc: true };
  window.addEventListener("keydown", (e) => {
    if (e.key === "r" || e.key === "R") view.enableWorldRc = !view.enableWorldRc;
  });

  // --- lil-gui: all world-RC properties, live-tunable via setParams. ---
  const gui = new GUI({ title: "World RC" });
  const wp = worldRc.params;
  // setParams(Object.assign(p, partial)) re-uploads every scalar + sun/sky vec4,
  // so passing the whole params object back applies whatever the GUI just mutated.
  const apply = () => worldRc.setParams(wp);

  gui.add(view, "enableWorldRc").name("enable world RC (R)").listen();

  // gridX/gridY/gridZ resize the probe atlases → full rebuild on release
  // (.onFinishChange, not .onChange, so we rebuild once when the slider settles).
  const rebuildDims = () => worldRc.rebuild({ gridX: wp.gridX, gridY: wp.gridY, gridZ: wp.gridZ });

  const grid = gui.addFolder("Probe grid");
  grid.add(wp, "cell0", 0.25, 6, 0.05).name("cell0 (world/probe)").onChange(apply);
  grid.add(wp, "cellZ", 0.25, 6, 0.05).name("cellZ (world/layer)").onChange(apply);
  grid.add(wp, "baseZ", -2, 12, 0.1).name("base Z (layer 0)").onChange(apply);
  grid.add(wp, "intervalStart", 0, 6, 0.05).name("interval start").onChange(apply);
  grid.add(wp, "intervalEnd", 0.25, 8, 0.05).name("base interval c0").onChange(apply);
  grid.add(wp, "gatherSteps", 4, 96, 1).name("gather steps").onChange(apply);
  // Atlas-size-defining dims → rebuild on release (destroy + recreate textures/shaders).
  grid.add(wp, "gridX", 16, 256, 16).name("grid X").onFinishChange(rebuildDims);
  grid.add(wp, "gridY", 16, 256, 16).name("grid Y").onFinishChange(rebuildDims);
  grid.add(wp, "gridZ", 1, 16, 1).name("grid Z (layers)").onFinishChange(rebuildDims);
  // dir tile side also defines atlas size; not part of this task's rebuild set.
  grid.add(wp, "dir0W").name("dir tile (fixed)").disable();

  const comp = gui.addFolder("Composite");
  comp.add(wp, "ambient", 0, 1, 0.01).onChange(apply);

  // Sun/sky drive the top-cascade miss term. SunLight (angle/enabled) is read live
  // every frame by the gather pass, so it needs no onChange; the colors do.
  const sun = gui.addFolder("Sun / Sky");
  sun.add(SunLight, "enabled").name("sun enabled");
  sun.add(SunLight, "angle", 0, Math.PI * 2, 0.01).name("sun angle");
  sun.addColor(wp, "sunColor").name("sun color").onChange(apply);
  sun.add(wp, "sunIntensity", 0, 5, 0.05).name("sun intensity").onChange(apply);
  sun.add(wp, "sunDistance", 0, 2, 0.01).name("sun softness").onChange(apply);
  sun.addColor(wp, "skyColor").name("sky color").onChange(apply);
  sun.add(wp, "skyMix", 0, 1, 0.01).name("sky mix").onChange(apply);

  // Numeric diagnostics: reads the probe atlas + final composite back to the CPU
  // and prints one paste-able report. Trigger with key L or the GUI button.
  const diag = createWorldRCDiagnostics({
    device,
    getProbeRad: () => rcTextures.probeRad,
    getProbeMerge: () => rcTextures.probeMerge,
    getWorldLit: () => worldRc.outputTexture,
    getRuntime: () => ({
      cell0: wp.cell0,
      baseZ: wp.baseZ,
      cellZ: wp.cellZ,
      gridX: wp.gridX,
      gridY: wp.gridY,
      gridZ: wp.gridZ,
      intervalEnd: wp.intervalEnd,
      gatherSteps: wp.gatherSteps,
      ambient: wp.ambient,
      sunEnabled: SunLight.enabled,
      sunAngle: SunLight.angle,
      sunIntensity: wp.sunIntensity,
      cameraX: cameraPosition.x,
      cameraY: cameraPosition.y,
      instanceCount: shapeSystem.sceneInstances.instanceCount,
    }),
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "l" || e.key === "L") diag.dump();
  });
  gui.add({ dump: () => diag.dump() }, "dump").name("dump diag (L)");

  // Surfels (debug): spawn on visible surfaces + overlay billboard dots.
  const surfelGui = { showSurfels: true };
  const sf = gui.addFolder("Surfels (debug)");
  sf.add(surfelGui, "showSurfels").name("show surfels");
  sf.add(surfel.params, "spawnChance", 0, 0.2, 0.0001)
    .name("spawn chance")
    .onChange(() => surfel.setParams(surfel.params));
  sf.add(surfel.params, "surfelRadius", 0.1, 3, 0.05)
    .name("surfel radius")
    .onChange(() => surfel.setParams(surfel.params));
  // Single DENSITY-shape knob: spawn-coverage distance as a multiple of radius.
  // cellSize, the spawn threshold and the (hysteresis) recycle threshold are all
  // derived from radius + this, so they can't be desynced. <1 = denser, >1 = sparser.
  sf.add(surfel.params, "coverage", 0.4, 2.0, 0.05)
    .name("coverage (×r)")
    .onChange(() => surfel.setParams(surfel.params));
  sf.add(surfel.params, "markerDecay", 0, 0.5, 0.005)
    .name("marker decay")
    .onChange(() => surfel.setParams(surfel.params));
  sf.add(surfel.params, "quadPx", 1, 20, 1)
    .name("dot size (px)")
    .onChange(() => surfel.setParams(surfel.params));
  sf.add({ clear: () => surfel.clear() }, "clear").name("clear surfels");
  sf.add(
    { log: () => surfel.readCount().then((n) => console.log("[surfels] count", n)) },
    "log",
  ).name("log count");

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

    // Recreate frame + RC textures + tick if the canvas was resized.
    if (canvas.width !== frameW || canvas.height !== frameH) {
      frame = createFrameTextures(device, canvas);
      frameW = canvas.width;
      frameH = canvas.height;
      frameTick = createFrameTick(
        { ...frame, canvas, device, background: [0.043, 0.051, 0.07, 1], getPixelRatio },
        ({ passEncoder }) => shapeSystem.drawShapes(passEncoder),
      );
      // World RC: recreate textures/pipelines (sceneInstances ref is stable).
      worldRc.recreate(canvas, frame.renderTexture, frame.depthTexture, frame.normalTexture);
      // Surfels: rebind the spawn compute to the new G-buffer textures.
      surfel.recreate(frame.depthTexture, frame.normalTexture);
    }

    execTransformSystem();
    shapeSystem.prepare();

    const encoder = device.createCommandEncoder();
    // Main pass -> renderTexture (raw albedo G-buffer).
    frameTick(encoder, delta);
    // World RC gather + merge + composite -> worldLitTexture (only when enabled).
    if (view.enableWorldRc) worldRc.run(encoder, delta);
    // Surfels (Stage B): clearGrid -> insert (rebuild hash grid from live set) ->
    // spawn (coverage-gated) -> recycle (age + push back over-covered/stale). Runs
    // regardless of RC state; converges the live count well below CAP.
    surfel.run(encoder, frameIndex);
    // Present the lit composite, or the raw unlit scene when RC is off.
    const presented = view.enableWorldRc ? worldRc.outputTexture : frame.renderTexture;
    // Debug overlay: draw the surfel dots ON TOP of the presented texture (loadOp
    // "load" keeps the lit scene; no depth, so they're not occluded — debug only).
    if (surfelGui.showSurfels) {
      const overlay = encoder.beginRenderPass({
        colorAttachments: [{ view: presented.createView(), loadOp: "load", storeOp: "store" }],
      });
      surfel.drawDebug(overlay);
      overlay.end();
    }
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
