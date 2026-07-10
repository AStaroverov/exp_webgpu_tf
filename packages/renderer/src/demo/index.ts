// renderer3d 2.5D true-3D-SDF demo — the harness/wiring only. The scenes themselves (world
// content + per-scene animation + per-scene GUI) live in src/demo/scenes/* behind the
// createScene registry; this file owns the engine wiring (createWorld → initWebGPU → frame
// textures → draw system → resize/transform systems → voxel GI → present), the shared voxel-GI
// GUI, and the frame loop.
//
// DEPTH CONVENTION — REVERSE-Z (NEAR=1 .. FAR=0): the draw pipeline compares
// depth "greater-equal" against a 0 clear; ResizeSystem.viewProjMatrix and the
// shader's frag_depth both follow it. See sdf.shader.ts / ResizeSystem.ts.

import GUI from "lil-gui";
import Stats from "stats-gl";
import { initWebGPU } from "../gpu.ts";
import {
  getGpuTimings,
  gpuTimerBeginFrame,
  gpuTimerPoll,
  gpuTimerResolve,
  initGpuTimer,
} from "../gpuTimer.ts";
import { createWorld } from "../ECS/world.ts";
import { createFrameTextures, createFrameTick } from "../WGSL/createFrame.ts";
import { createPresent } from "../WGSL/createPresent.ts";
import { createDrawShapeSystem } from "../ECS/Systems/SDFSystem/createDrawShapeSystem.ts";
import { createVoxelSystem } from "../ECS/Systems/Lighting/createVoxelSystem.ts";
import {
  GI_QUALITY_PRESETS,
  type GIQuality,
} from "../ECS/Systems/Lighting/core/voxelConfig.ts";
import { createLightEmitterSystem } from "../ECS/Systems/Lighting/lights/createLightEmitterSystem.ts";
import { SunLight } from "../ECS/Systems/SunLight.ts";
import { createTransformSystem } from "../ECS/Systems/TransformSystem.ts";
import {
  cameraAzimuth,
  cameraElevation,
  cameraPosition,
  cameraZoom,
  createResizeSystem,
  setCameraElevation,
  setCameraPosition,
} from "../ECS/Systems/ResizeSystem.ts";
import { createScene, SCENE_OPTIONS, type SceneName } from "./scenes/index.ts";
import { perfToggles } from "./scenes/perfToggles.ts";

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
  // Per-pass GPU profiler (timestamp queries). False when the adapter lacks the feature — the
  // instrumented passes then run untagged with zero overhead.
  const gpuTimerOn = initGpuTimer(device);

  const world = createWorld();

  // Scene selection (picked live from the GUI; persisted in localStorage, applied on reload).
  const savedScene = localStorage.getItem("demo.scene");
  const SCENE: SceneName = (SCENE_OPTIONS as readonly string[]).includes(savedScene ?? "")
    ? (savedScene as SceneName)
    : "showcase";
  // Both perf scenes drive the same GPU-cost harness (per-pass toggles + serialized timing).
  const PERF = SCENE === "perf" || SCENE === "perf2";

  // Spawn the scene's world content; keep its hooks (animate / setupGUI / camera framing).
  const scene = createScene(SCENE, world);

  // --- Systems ---
  const execTransformSystem = createTransformSystem(world, stubChildren, -1);
  const shapeSystem = createDrawShapeSystem({ world, device });
  const present = createPresent(device, context);

  setCameraPosition(0, 0);
  // pixels per world unit; smaller shows more world. Scenes that spread wide/tall override it.
  cameraZoom.value = scene.cameraZoom ?? 14;

  let frame = createFrameTextures(device, canvas);
  let frameW = canvas.width;
  let frameH = canvas.height;
  let frameTick = createFrameTick(
    { ...frame, canvas, device, background: [0.043, 0.051, 0.07, 1], getPixelRatio },
    ({ passEncoder }) => shapeSystem.drawShapes(passEncoder),
  );

  // Voxel GI system: voxelize the scene, build the radiance pyramid, cone-gather + composite.
  const voxel = createVoxelSystem({
    device,
    canvas,
    sceneInstances: shapeSystem.sceneInstances,
    depthTexture: frame.depthTexture,
    normalTexture: frame.normalTexture,
    albedoTexture: frame.renderTexture,
    emissionTexture: frame.emissionTexture,
  });

  const gui = new GUI({ title: "Voxel" });

  // Per-pass GPU timings (ms, EMA) from timestamp queries — the ground truth for every perf
  // decision. Rows follow the gpuSpan labels; same-label passes (mips, aniso levels, the voxelize
  // trio) are pre-summed. NOTE: Chrome quantizes timestamps to 100 µs by default — launch with
  // --enable-webgpu-developer-features for µs precision on sub-0.1ms passes.
  const timingsFolder = gui.addFolder("GPU timings (ms)");
  const gpuStats: Record<string, number> = {
    draw: 0,
    sunDepth: 0,
    voxelize: 0,
    mips: 0,
    anisoBase: 0,
    anisoMips: 0,
    probeGather: 0,
    coneResolve: 0,
    coneTemporal: 0,
    composite: 0,
    total: 0, // sum of the rows — INFLATED when pass windows overlap (see frameSpan)
    frameSpan: 0, // first begin → last end: the frame's true GPU window (compare vs stats-gl)
  };
  if (gpuTimerOn) {
    for (const key of Object.keys(gpuStats)) {
      timingsFolder.add(gpuStats, key).decimals(3).disable().listen();
    }
  } else {
    timingsFolder.add({ status: "timestamp-query N/A" }, "status").disable();
  }
  // Serialize CPU↔GPU per frame (await onSubmittedWorkDone) — DIAGNOSTIC only: gives a clean
  // wall-clock GPU number for the old toggle-delta workflow, but caps fps at CPU+GPU instead of
  // max(CPU, GPU). OFF by default — the timestamp table above supersedes it.
  const serializeCfg = { serializeGpu: false };
  timingsFolder.add(serializeCfg, "serializeGpu").name("serialize GPU (diag)");
  timingsFolder.close();
  const updateGpuStats = () => {
    if (!gpuTimerOn) return;
    let total = 0;
    for (const [label, ms] of getGpuTimings()) {
      if (label in gpuStats) gpuStats[label] = ms;
      if (label !== "frameSpan") total += ms;
    }
    gpuStats.total = total;
  };

  gui
    .add({ scene: SCENE }, "scene", SCENE_OPTIONS as unknown as string[])
    .name("scene")
    .onChange((v: string) => {
      localStorage.setItem("demo.scene", v);
      location.reload();
    });

  // Graininess: voxel size in world units. Smaller = finer = more voxels. Rebuilds the
  // 3D textures on release (.onFinishChange, so it rebuilds once when the slider settles).
  // The displayed dims controller reflects the resulting per-axis voxel counts.
  // NOTE: with "auto cell (zoom)" ON the ladder owns cellSize (the live readout below tracks it);
  // touching the manual slider switches auto OFF (setCellSize is an override).
  const voxCfg = { cellSize: voxel.cellSize };
  const dimsLabel = { dims: `${voxel.dims.x}×${voxel.dims.y}×${voxel.dims.z}` };
  const dimsCtl = gui.add(dimsLabel, "dims").name("voxel dims").disable();
  gui
    .add(voxCfg, "cellSize", 0.125, 2, 0.025)
    .name("voxel size (graininess)")
    .onFinishChange((cs: number) => {
      voxel.setCellSize(cs);
      dimsLabel.dims = `${voxel.dims.x}×${voxel.dims.y}×${voxel.dims.z}`;
      dimsCtl.updateDisplay();
      autoCellCfg.auto = voxel.autoCell; // slider overrides → auto flips off; reflect it
      autoCellCtl.updateDisplay();
    });
  // Zoom ladder: cellSize follows the zoom in discrete ×2 steps so the box always covers the
  // screen (+ off-screen light margin) at constant texture dims. The readout shows the live cell.
  const autoCellCfg = { auto: voxel.autoCell };
  const autoCellCtl = gui
    .add(autoCellCfg, "auto")
    .name("auto cell (zoom)")
    .onChange((on: boolean) => voxel.setAutoCell(on));
  const cellLive = {
    get cell() {
      return voxel.cellSize;
    },
  };
  gui.add(cellLive, "cell").name("cell (live)").disable().listen();

  // A/B for the camera-following voxel box: off = the box freezes at its current origin (the old
  // fixed-world-box behavior), so panning past its edge shows the no-GI falloff again.
  const followCfg = { follow: voxel.followCamera, snapCells: voxel.gridSnapCells };
  gui
    .add(followCfg, "follow")
    .name("grid follows camera")
    .onChange((on: boolean) => voxel.setFollowCamera(on));
  // Snap quantum trade-off: bigger = more mip levels world-locked while panning (less far-field
  // pattern re-forming on a snap step), but the coverage box steps coarser. 16 locks iso mip ≤ 4.
  gui
    .add(followCfg, "snapCells", [4, 8, 16, 32])
    .name("grid snap (voxels)")
    .onChange((c: number) => voxel.setGridSnapCells(c));

  // Sun toggle is read live by the draw pass; keep it exposed for the raw view.
  gui.add(SunLight, "enabled").name("sun enabled");
  gui.add(SunLight, "angle", 0, Math.PI * 2, 0.01).name("sun angle");
  gui.add(SunLight, "elevation", 0, Math.PI / 2, 0.01).name("sun elevation");
  gui.add(SunLight, "intensity", 0, 5, 0.05).name("sun intensity");
  gui.addColor(SunLight, "color", 1).name("sun color"); // rgbScale=1 → array is 0..1 floats

  // Cone GI: the screen-probe RESOLVE (the probe SH carries fill/bounce AND emitter light) + the
  // short per-pixel AO cones. No aimed cones here anymore — the emitter knobs live in the
  // "Screen probe GI" folder (they bake into the probe gather).
  // GI QUALITY PRESET: one switch over the perf-relevant baked knobs (probe tile, cone/AO/aimed
  // budgets, reach, aniso, temporal hystereses, resolve radius) + the cone-pass resolution.
  // Artistic tuning (strengths, exposure, sun…) is untouched. Applies + rebuilds immediately; the
  // individual knobs below stay usable to deviate from a preset. (coneResCfg is declared below —
  // the closure only runs on user input, long after setup.)
  const qualityCfg = { quality: "medium" as GIQuality };
  gui
    .add(qualityCfg, "quality", ["low", "medium", "high"])
    .name("GI quality")
    .onChange((q: GIQuality) => {
      const preset = GI_QUALITY_PRESETS[q];
      Object.assign(voxel.config, preset.config);
      coneResCfg.scale = preset.coneScale;
      voxel.setConeScale(preset.coneScale);
      voxel.rebuild();
      // The preset mutated voxel.config fields other controllers are bound to — refresh them all.
      gui.controllersRecursive().forEach((c) => c.updateDisplay());
    });

  const coneFolder = gui.addFolder("Cone GI");
  // Baked-config controls recompile the GI shaders on release (onFinishChange), not per drag tick.
  const rebuild = () => voxel.rebuild();
  coneFolder
    .add(voxel.config, "aperture", 0.1, 1.5, 0.01)
    .name("aperture (lower=sharper)")
    .onFinishChange(rebuild);
  coneFolder.add(voxel.config, "maxDist", 1, 64, 0.5).name("cone reach").onFinishChange(rebuild);
  coneFolder
    .add(voxel.config, "normalBias", 0, 2, 0.01)
    .name("normal bias")
    .onFinishChange(rebuild);
  coneFolder
    .add(voxel.config, "giStrength", 0, 4, 0.05)
    .name("GI strength (bounce)")
    .onFinishChange(rebuild);
  // Cone-pass resolution: 2 = half-res (¼ pixels), 4 = quarter-res (1/16), 8 = eighth-res (1/64).
  // The biggest perf lever for heavy scenes — lower res blurs the GI but the bilateral upsample
  // keeps edges crisp.
  const coneResCfg = { scale: voxel.coneScale };
  coneFolder
    .add(coneResCfg, "scale", { "half-res (2)": 2, "quarter-res (4)": 4, "eighth-res (8)": 8 })
    .name("cone resolution")
    .onChange((s: number) => voxel.setConeScale(s));
  // Anisotropic voxels: directional far-field volumes (anti-leak) vs the plain isotropic pyramid.
  // BAKED (rebuild) — flip it to see light stop bleeding through thin occluders.
  coneFolder.add(voxel.config, "anisoMode").name("anisotropic voxels").onFinishChange(rebuild);

  // Screen-probe GI: surface-anchored probes (one per tile) that supply the diffuse fill/bounce
  // AND the aimed emitter cones (traced once per probe — the emitter knobs below bake into the
  // probe gather). conesPerProbe is the bounce quality (probes run at low res, once per frame →
  // afford many); aoConeCount/aoReach are the SHORT per-pixel contact-AO cones (the .a/visibility
  // term, still in the cone pass).
  const probeFolder = gui.addFolder("Screen probe GI");
  // Emitter DIRECT strength: multiplier on the aimed-cone direct light. Raise it to let a
  // bright emitter overpower the sun (e.g. fill the sun-shadow it casts under itself).
  probeFolder
    .add(voxel.config, "emitterDirect", 0, 8, 0.1)
    .name("emitter direct strength")
    .onFinishChange(rebuild);
  // Distance falloff for emitter direct light: 0 = flat (sun-like, hard rim), 1 = standard 1/d².
  probeFolder
    .add(voxel.config, "emitterFalloff", 0, 4, 0.05)
    .name("emitter falloff")
    .onFinishChange(rebuild);
  // Aimed-cone march budget: fewer steps = cheaper, but shorter/coarser emitter shadows (and
  // possible light leak through thin occluders).
  probeFolder.add(voxel.config, "aimedSteps", 8, 64, 1).name("aimed steps").onFinishChange(rebuild);
  // Early-out opacity: <1 lets a near-opaque aimed cone stop before its full budget (saves the tail
  // when the light is blocked). 1 = no early cut (sharpest shadow).
  probeFolder
    .add(voxel.config, "aimedAlphaCut", 0.5, 1, 0.01)
    .name("aimed alpha cut")
    .onFinishChange(rebuild);
  // Aimed cones per probe per FRAME: with more live lights in the probe's cluster cell it
  // round-robins a window of this many (energy-rescaled; the temporal history integrates the
  // rest — needs hysteresis > 0). The knob = the emitter cost ceiling per probe.
  probeFolder
    .add(voxel.config, "aimedPerFrame", 1, 16, 1)
    .name("aimed cones / frame")
    .onFinishChange(rebuild);
  // Clustered light culling: cluster cell size (in voxels per axis) + max lights recorded per
  // cell (overflow is dropped for that cell).
  probeFolder
    .add(voxel.config, "clusterDiv", [2, 4, 8, 16, 32])
    .name("light cluster size (voxels)")
    .onFinishChange(rebuild);
  probeFolder
    .add(voxel.config, "clusterCap", 4, 64, 1)
    .name("lights / cluster cap")
    .onFinishChange(rebuild);
  // SH-L1 saturates ~16 cones, so higher values only cut noise (no detail) — keep this low.
  probeFolder
    .add(voxel.config, "conesPerProbe", [8, 16, 32, 64, 128])
    .name("cones / probe")
    .onFinishChange(rebuild);
  probeFolder
    .add(voxel.config, "aoConeCount", 0, 8, 1)
    .name("AO cones (contact)")
    .onFinishChange(rebuild);
  probeFolder.add(voxel.config, "aoReach", 21, 16, 0.5).name("AO reach").onFinishChange(rebuild);
  // Screen-probe tile (full-res px / probe): smaller = finer probe grid = sharper fill but more
  // gather cost. BAKED (rebuild — also recreates the atlas it sizes).
  probeFolder
    .add(voxel.config, "screenProbeTile", [2, 4, 8, 16, 24, 32, 48, 64])
    .name("probe tile (px)")
    .onFinishChange(rebuild);
  // Bilateral resolve weights (BAKED — rebuild): normalPow = normal-similarity sharpness (higher =
  // stricter across differing normals); planeK = plane-reject threshold × local probe spacing
  // (lower = stricter across depth steps → less bleed but more disocclusion fallback).
  probeFolder
    .add(voxel.config, "spNormalPow", 0.5, 8, 0.5)
    .name("resolve: normal pow")
    .onFinishChange(rebuild);
  probeFolder
    .add(voxel.config, "spPlaneK", 0.25, 4, 0.25)
    .name("resolve: plane K")
    .onFinishChange(rebuild);
  // Unified-resolve support radius (in local probe pitches): the smooth screen kernel that weights
  // EVERY probe tapers to zero at this distance. Bigger = smoother/wider fill (also helps a distant
  // object seen by few probes); smaller = more local detail. BAKED (rebuild).
  probeFolder
    .add(voxel.config, "resolveRadius", 0.5, 3, 0.25)
    .name("resolve radius (tiles)")
    .onFinishChange(rebuild);
  // STAGE 3: temporal accumulation on the probe atlas — the history weight of the per-probe SH
  // blend. 0 = OFF (fresh-only — the A/B + rollback); ~0.85–0.9 amortizes the gather across frames
  // (per-frame golden-angle cone rotation integrates back to an effective 2–4× cone budget), so
  // cones/probe can drop to 4–8. BAKED (rebuild) — 0 makes the whole history block dead code.
  probeFolder
    .add(voxel.config, "temporalHysteresis", 0, 0.95, 0.05)
    .name("temporal hysteresis")
    .onFinishChange(rebuild);
  // POINT C: the temporal filter on the RESOLVED cone output (reproject + 3×3 neighborhood clamp +
  // blend). Catches the resolve-level noise the probe history can't (probe-set churn at ring/refine
  // boundaries, per-pixel AO jitter, motion-time freshness) — 0 = off (exact passthrough A/B).
  // BAKED (rebuild).
  probeFolder
    .add(voxel.config, "coneTemporalHysteresis", 0, 0.95, 0.05)
    .name("resolve temporal (C)")
    .onFinishChange(rebuild);


  // Composite (Layer 4): the final lit image. Sun controls above feed it via SunLight;
  // cone giStrength bakes into the indirect term. Only the ambient floor lives here.
  const compositeFolder = gui.addFolder("Composite");
  compositeFolder
    .add(voxel.config, "ambient", 0, 0.5, 0.01)
    .name("composite ambient")
    .onFinishChange(rebuild);
  // HDR exposure before the ACES tonemap. Raise for a brighter image; highlights roll off instead
  // of clipping to flat white.
  compositeFolder
    .add(voxel.config, "exposure", 0.1, 4, 0.05)
    .name("exposure")
    .onFinishChange(rebuild);
  // Sun-shadow penumbra: the PCF filter widens as the sun intensity drops below 1, so a dimmer sun
  // casts a softer, wider shadow edge. 0 = always crisp; higher = stronger softening when sun < 1.
  compositeFolder
    .add(voxel.config, "penumbra", 0, 12, 0.5)
    .name("penumbra (sun-dim)")
    .onFinishChange(rebuild);
  // Base sun-shadow PCF radius applied even at full sun → smooths the shadow-map texel staircase.
  // 1 = near-hard (old). The sun frustum also auto-fits the camera view, so steps shrink on zoom.
  compositeFolder
    .add(voxel.config, "shadowBaseSpread", 1, 6, 0.25)
    .name("shadow softness")
    .onFinishChange(rebuild);

  // Scene-specific GUI (perf toggles / emitter controls / animation switches / …).
  scene.setupGUI?.(gui);

  // Auto-discover scene emitters → cone importance-sampling lights each frame.
  const updateLights = createLightEmitterSystem(world, voxel);

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

  // --- WASD / arrow-key panning: moves the camera look-at target across the ground plane in
  // SCREEN-relative directions (W = away from the viewer, derived from the current azimuth).
  // Exists to validate the camera-following voxel box: pan past the old fixed-box edge and watch
  // GI coverage travel with the camera (and stay crawl-free thanks to the origin snap).
  const heldKeys = new Set<string>();
  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement) return; // don't pan while typing in the GUI
    heldKeys.add(e.code);
  });
  window.addEventListener("keyup", (e) => heldKeys.delete(e.code));
  function panCamera(deltaMs: number) {
    const dx =
      (heldKeys.has("KeyD") || heldKeys.has("ArrowRight") ? 1 : 0) -
      (heldKeys.has("KeyA") || heldKeys.has("ArrowLeft") ? 1 : 0);
    const dy =
      (heldKeys.has("KeyW") || heldKeys.has("ArrowUp") ? 1 : 0) -
      (heldKeys.has("KeyS") || heldKeys.has("ArrowDown") ? 1 : 0);
    if (dx === 0 && dy === 0) return;
    // Ground-plane camera basis from the azimuth: forward = away from the viewer, right = screen
    // right (matches the lookAt basis in ResizeSystem). Speed scales inversely with zoom so the
    // on-screen pan rate feels constant.
    const a = (cameraAzimuth.value * Math.PI) / 180;
    const fx = -Math.cos(a);
    const fy = -Math.sin(a);
    const speed = (150 / cameraZoom.value) * (deltaMs / 1000);
    setCameraPosition(
      cameraPosition.x + (fx * dy + fy * dx) * speed,
      cameraPosition.y + (fy * dy - fx * dx) * speed,
    );
  }

  // stats-gl overlay: FPS + CPU come from begin()/end() (no init needed). stats-gl's NATIVE
  // GPU timer needs a WebGL2 context or a three.js renderer — we have neither (raw WebGPU),
  // so we feed our own GPU ms (onSubmittedWorkDone, below) into a custom panel instead.
  const stats = new Stats({ trackGPU: false, horizontal: true, precision: 2 });
  document.body.appendChild(stats.dom);
  const gpuPanel = stats.addPanel(new Stats.Panel("GPU ms", "#ff8", "#221"));
  let gpuMsMax = 1;

  let last = performance.now();
  let gpuMsEMA = 0;
  async function loop(now: number) {
    stats.begin();
    const delta = Math.min(now - last, 16.6667);
    last = now;

    // Update camera + canvas size first, so prepare() uploads current uniforms
    // and the resize check below sees this frame's dimensions.
    panCamera(delta);
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

    // Drive the scene's dynamic objects, then push transforms → instance buffers.
    scene.animate?.(now);
    // Auto-discover emitters (positions are now current) → cone importance-sampling lights.
    updateLights();
    execTransformSystem();
    shapeSystem.prepare();

    gpuTimerBeginFrame(); // reset the span allocator before any pass is encoded
    const encoder = device.createCommandEncoder();
    if (PERF) {
      // Perf harness: every pass runs purely by its toggle (skipped passes just leave their
      // textures stale — valid GPU work, no crash). Always presents the composite output, so
      // toggling a pass changes ONLY that pass's GPU work → the gpuMs delta attributes its
      // cost. The full chain is voxelize → mips → cone → composite (+ the SDF draw G-buffer).
      // sunDepth runs FIRST so model A's voxelize can sample the sun shadow map (and
      // buildSunViewProj uploads the matrix to voxelize + composite). If sunDepth is toggled OFF
      // while voxelize is ON in model A, voxelize samples a STALE sun depth map — acceptable for
      // a cost harness (the binding is always valid; no crash).
      if (perfToggles.draw) frameTick(encoder, delta);
      if (perfToggles.sunDepth) voxel.sunDepth(encoder); // sun-POV depth feeding voxelize (A) + composite
      if (perfToggles.voxelize) voxel.voxelize(encoder);
      if (perfToggles.mips) voxel.mips(encoder);
      if (perfToggles.anisoBase) voxel.anisoBase(encoder);
      if (perfToggles.anisoMips) voxel.anisoMips(encoder);
      // Screen-probe gather (the diffuse fill source): reads the radiance pyramid (after mips),
      // must precede cone (which resolves it).
      if (perfToggles.screenProbe) voxel.gatherProbes(encoder);
      if (perfToggles.cone) voxel.cone(encoder);
      if (perfToggles.composite) voxel.composite(encoder);
      present(encoder, voxel.compositeOutputTexture);
    } else {
      // Final lit image: SDF G-buffer draw → the full voxel-GI scenario (voxel.renderFrame,
      // load-bearing order — sunDepth is gated on the sun internally) → present.
      frameTick(encoder, delta);
      voxel.renderFrame(encoder);
      present(encoder, voxel.compositeOutputTexture);
    }

    // Per-pass timestamps: resolve this frame's spans + stage the readback copy.
    gpuTimerResolve(encoder);

    const tSubmit = performance.now();
    device.queue.submit([encoder.finish()]);
    gpuTimerPoll(); // kick the async map of the staged timestamp copy
    updateGpuStats(); // refresh the GUI rows from the latest completed readback
    if (serializeCfg.serializeGpu) {
      // DIAGNOSTIC mode: wait for THIS frame's GPU work before encoding the next. Removes the
      // cross-frame queue backlog so a wall-clock submit→done delta is a clean single-frame number
      // — but it SERIALIZES CPU and GPU (frame = CPU encode + GPU execute, back to back), capping
      // fps at the SUM instead of the max. Never how a shipping loop runs.
      await device.queue.onSubmittedWorkDone();
      const dt = performance.now() - tSubmit;
      gpuMsEMA = gpuMsEMA ? gpuMsEMA * 0.8 + dt * 0.2 : dt;
      gpuMsMax = Math.max(gpuMsMax, gpuMsEMA);
      gpuPanel.update(gpuMsEMA, gpuMsMax);
    } else {
      // PIPELINED (default): CPU encodes frame N+1 while the GPU draws frame N — fps = max(CPU,
      // GPU), not the sum. The GPU panel shows the timestamp-derived frameSpan (the honest GPU
      // frame time; no wall-clock GPU number exists without serializing).
      gpuMsEMA = gpuStats.frameSpan;
      gpuMsMax = Math.max(gpuMsMax, gpuMsEMA);
      gpuPanel.update(gpuMsEMA, gpuMsMax);
    }

    // CPU/FPS frame bracket for stats-gl (GPU panel is fed by the timer above).
    stats.end();
    stats.update();

    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f88;padding:20px">${(err as Error)?.stack ?? err}</pre>`;
  console.error(err);
});
