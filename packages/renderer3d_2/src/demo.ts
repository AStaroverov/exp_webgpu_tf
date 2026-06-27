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
import Stats from "stats-gl";
import { hasComponent, query } from "bitecs";
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
  const { LocalTransform, LightEmitter, Shape, Height, Color } = getRenderComponents(world);

  // Scene selection:
  //   "emitter"  — ground + box occluder + a GUI-movable/resizable emitter sphere.
  //   "simple"   — fixed minimal scene (first-bug diagnosis).
  //   "showcase" — one of every shape kind + several lights.
  const SCENE = "perf2" as "emitter" | "showcase" | "final" | "perf" | "perf2";
  // Both perf scenes drive the same GPU-cost harness (per-pass toggles + serialized timing).
  const PERF = SCENE === "perf" || SCENE === "perf2";

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

  // Perf-test scene: a dense instance grid + per-pass toggles so the cost of each pass can
  // be isolated by reading the GPU-time delta when a layer is switched off. `stats` is the
  // live readout (rAF fps + onSubmittedWorkDone GPU ms, the latter NOT vsync-capped).
  const perf = {
    animate: true,
    draw: true, // SDF G-buffer draw pass (frameTick)
    voxelize: true, // scene → 3D voxel textures
    mips: true, // radiance mip pyramid
    probe: true, // irradiance-probe SH volume (fill/bounce)
    cone: true, // N-cone GI gather (half-res)
    sunDepth: true, // sun shadow-map depth pass (sun-POV SDF depth)
    composite: true, // final lit image
  };
  const perfEntities: number[] = []; // ids of the grid instances (spun by animatePerf / animatePerf2)

  // perf2 scene: 5 stacked Z-layers of small MOVING shapes (150/layer = 750) with emitters
  // weaving through the gaps. The target workload — many small dynamic objects re-voxelized every
  // frame. Geometry is shared between the spawn block and animatePerf2 (homes drift around these).
  const PERF2 = { layers: 5, cols: 15, rows: 10, spanX: 42, spanY: 28, z0: 1.0, dz: 2.4 };
  const perf2Homes: number[] = []; // x,y,z per shape, parallel to perfEntities (the orbit anchors)
  const perf2Emitters: number[] = []; // ids of the emitters flying between the layers

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

  // --- Perf scene: a dense grid of mixed shapes to stress voxelize (O(voxels×instances))
  //     and the SDF draw (per-fragment march + overdraw), plus a few emitters. Heavy enough
  //     to push GPU time well above the inspector's ~4 ms floor so pass toggles read clearly.
  if (SCENE === "perf") {
    SunLight.enabled = true;
    SunLight.angle = 2.4;
    SunLight.elevation = 0.95;
    SunLight.intensity = 0.9;
    SunLight.color = [1.0, 0.93, 0.82];

    createRectangle(world, {
      x: 0,
      y: 0,
      z: -0.4,
      width: 80,
      height: 80,
      depth: 0.4,
      color: [0.2, 0.2, 0.22, 1],
    });

    // N×N grid of alternating box / sphere / cylinder (covers all three march costs).
    const N = 11;
    const spacing = 4.5;
    const half = ((N - 1) * spacing) / 2;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const x = i * spacing - half;
        const y = j * spacing - half;
        const k = (i + j) % 3;
        let id: number;
        if (k === 0) {
          id = createRectangle(world, {
            x,
            y,
            z: 0,
            width: 2,
            height: 2,
            depth: 3,
            color: [0.8, 0.5, 0.4, 1],
          });
        } else if (k === 1) {
          id = createSphere(world, { x, y, z: 0, radius: 1.3, color: [0.5, 0.7, 0.85, 1] });
        } else {
          id = createCircle(world, {
            x,
            y,
            z: 0,
            radius: 1.2,
            height: 3,
            color: [0.6, 0.8, 0.6, 1],
          });
        }
        perfEntities.push(id);
      }
    }

    // Three emitters at mid height (radiance sources for the cone gather).
    const pe1 = createSphere(world, { x: -half, y: 0, z: 3, radius: 1, color: [1, 0.6, 0.2, 1] });
    LightEmitter.addComponent(world, pe1, 6.0);
    const pe2 = createSphere(world, { x: half, y: 0, z: 3, radius: 1, color: [0.3, 0.6, 1, 1] });
    LightEmitter.addComponent(world, pe2, 6.0);
    const pe3 = createSphere(world, { x: 0, y: half, z: 3, radius: 1, color: [0.8, 1, 0.5, 1] });
    LightEmitter.addComponent(world, pe3, 6.0);
  }

  // --- perf2 scene: 5 layers × 150 small movers + emitters weaving between the layers. ---
  if (SCENE === "perf2") {
    SunLight.enabled = true;
    SunLight.angle = 2.4;
    SunLight.elevation = 0.95;
    SunLight.intensity = 0.9;
    SunLight.color = [1.0, 0.93, 0.82];

    // Ground slab to catch the cast shadows + bounce from the stacked movers above.
    createRectangle(world, {
      x: 0,
      y: 0,
      z: -0.4,
      width: 64,
      height: 64,
      depth: 0.4,
      color: [0.2, 0.2, 0.22, 1],
    });

    // Per-layer tint so the 5 stacked sheets read apart visually.
    const layerTint: [number, number, number][] = [
      [0.85, 0.5, 0.4],
      [0.5, 0.7, 0.85],
      [0.6, 0.82, 0.55],
      [0.8, 0.6, 0.85],
      [0.85, 0.8, 0.45],
    ];

    for (let L = 0; L < PERF2.layers; L++) {
      const z = PERF2.z0 + L * PERF2.dz;
      const tint = layerTint[L % layerTint.length];
      for (let r = 0; r < PERF2.rows; r++) {
        for (let c = 0; c < PERF2.cols; c++) {
          const x = (PERF2.cols === 1 ? 0 : c / (PERF2.cols - 1) - 0.5) * PERF2.spanX;
          const y = (PERF2.rows === 1 ? 0 : r / (PERF2.rows - 1) - 0.5) * PERF2.spanY;
          const idx = (L * PERF2.rows + r) * PERF2.cols + c;
          const color: [number, number, number, number] = [tint[0], tint[1], tint[2], 1];
          const k = idx % 3;
          let id: number;
          if (k === 0) {
            id = createRectangle(world, { x, y, z, width: 1.1, height: 1.1, depth: 1.1, color });
          } else if (k === 1) {
            id = createSphere(world, { x, y, z, radius: 0.6, color });
          } else {
            id = createCircle(world, { x, y, z, radius: 0.55, height: 1.1, color });
          }
          perfEntities.push(id);
          perf2Homes.push(x, y, z);
        }
      }
    }

    // Emitters that fly through the gaps between layers (animated in animatePerf2). Distinct hues;
    // capped at the cone pass's 8-light limit, so keep this ≤ 8.
    const emColors: [number, number, number, number][] = [
      [1.0, 0.6, 0.2, 1],
      [0.3, 0.6, 1.0, 1],
      [0.7, 1.0, 0.4, 1],
      [1.0, 0.4, 0.8, 1],
      [0.95, 0.95, 0.9, 1],
    ];
    for (let e = 0; e < emColors.length; e++) {
      const id = createSphere(world, {
        x: 0,
        y: 0,
        z: PERF2.z0 + e * PERF2.dz * 0.8,
        radius: 0.8,
        color: emColors[e],
      });
      LightEmitter.addComponent(world, id, 8.0);
      perf2Emitters.push(id);
    }
  }

  // --- Systems ---
  const execTransformSystem = createTransformSystem(world, stubChildren);
  const shapeSystem = createDrawShapeSystem({ world, device });
  const present = createPresent(device, context);

  setCameraPosition(0, 0);
  // pixels per world unit; smaller shows more world. perf2 stacks 5 layers tall over a wide span,
  // so pull back to frame the whole thing.
  cameraZoom.value = SCENE === "perf2" ? 9 : 14;

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

  // Graininess: voxel size in world units. Smaller = finer = more voxels. Rebuilds the
  // 3D textures on release (.onFinishChange, so it rebuilds once when the slider settles).
  // The displayed dims controller reflects the resulting per-axis voxel counts.
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
    });

  // Sun toggle is read live by the draw pass; keep it exposed for the raw view.
  gui.add(SunLight, "enabled").name("sun enabled");
  gui.add(SunLight, "angle", 0, Math.PI * 2, 0.01).name("sun angle");
  gui.add(SunLight, "elevation", 0, Math.PI / 2, 0.01).name("sun elevation");
  gui.add(SunLight, "intensity", 0, 5, 0.05).name("sun intensity");
  gui.addColor(SunLight, "color", 1).name("sun color"); // rgbScale=1 → array is 0..1 floats

  // Cone GI: per-pixel AIMED emitter cones (sharp direct + shadow). The fill/bounce now comes
  // from the probe SH volume (see Probe GI folder); coneCount only sets the fill/bounce BLEND
  // weight (= count/2). Read live each frame by cone().
  const coneFolder = gui.addFolder("Cone GI");
  // Emitter DIRECT strength: multiplier on the summed aimed-cone direct light. Raise it to let a
  // bright emitter overpower the sun (e.g. fill the sun-shadow it casts under itself).
  coneFolder.add(voxel.coneParams, "emitterDirect", 0, 8, 0.1).name("emitter direct strength");
  // Distance falloff for emitter direct light: 0 = flat (sun-like, hard rim), 1 = standard 1/d².
  coneFolder.add(voxel.coneParams, "emitterFalloff", 0, 4, 0.05).name("emitter falloff");
  coneFolder.add(voxel.coneParams, "aperture", 0.1, 1.5, 0.01).name("aperture (lower=sharper)");
  coneFolder.add(voxel.coneParams, "maxDist", 1, 64, 0.5).name("cone reach");
  coneFolder.add(voxel.coneParams, "normalBias", 0, 2, 0.01).name("normal bias");
  coneFolder.add(voxel.coneParams, "giStrength", 0, 4, 0.05).name("GI strength (bounce)");

  // Probe GI: the low-res irradiance-probe volume that replaced the per-pixel fill hemisphere.
  // conesPerProbe is the bounce quality (probes run at low res, once per frame → afford many);
  // aoConeCount/aoReach are the SHORT per-pixel contact-AO cones (the .a/visibility term).
  const probeFolder = gui.addFolder("Probe GI");
  // SH-L1 saturates ~16 cones, so higher values only cut noise (no detail) — keep this low.
  probeFolder.add(voxel.probeParams, "conesPerProbe", [8, 16, 32, 64, 128]).name("cones / probe");
  probeFolder.add(voxel.probeParams, "aoConeCount", 0, 8, 1).name("AO cones (contact)");
  probeFolder.add(voxel.probeParams, "aoReach", 1, 16, 0.5).name("AO reach");

  // Composite (Layer 4): the final lit image. Sun controls above feed it via SunLight;
  // cone giStrength bakes into the indirect term. Only the ambient floor lives here.
  const compositeFolder = gui.addFolder("Composite");
  compositeFolder.add(voxel.compositeParams, "ambient", 0, 0.5, 0.01).name("composite ambient");
  // HDR exposure before the ACES tonemap. Raise for a brighter image; highlights roll off instead
  // of clipping to flat white.
  compositeFolder.add(voxel.compositeParams, "exposure", 0.1, 4, 0.05).name("exposure");
  // Sun-shadow penumbra: the PCF filter widens as the sun intensity drops below 1, so a dimmer sun
  // casts a softer, wider shadow edge. 0 = always crisp; higher = stronger softening when sun < 1.
  compositeFolder.add(voxel.compositeParams, "penumbra", 0, 12, 0.5).name("penumbra (sun-dim)");

  // Perf folder: live fps / GPU-ms readout + per-pass toggles. Read the GPU-ms delta when a
  // toggle flips to attribute cost to that pass (gpuMs comes from onSubmittedWorkDone, so it
  // is NOT capped by vsync the way the rAF fps is). Only shown for the perf scene.
  if (PERF) {
    const pf = gui.addFolder("Perf");
    pf.add(perf, "animate").name("animate grid");
    pf.add(perf, "draw").name("1· SDF draw pass");
    pf.add(perf, "voxelize").name("2· voxelize");
    pf.add(perf, "mips").name("3· mips");
    pf.add(perf, "probe").name("4· probe GI (SH)");
    pf.add(perf, "cone").name("5· cone GI");
    pf.add(perf, "sunDepth").name("6· sun shadow-map pass");
    pf.add(perf, "composite").name("7· composite");
    pf.open();
  }

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

  // Auto-discover every emitter from the ECS each frame and hand the cone pass their world
  // centers + radius for importance sampling — NO manual light list. Any entity with a
  // LightEmitter is a light, treated identically. center = transform translation + (0,0,radius),
  // radius ≈ half-height (the sphere radius for emitter spheres). Capped at 8 by setLights.
  const emitterLightsFlat: number[] = [];
  const emitterColorsFlat: number[] = []; // r,g,b,intensity per light (analytic-direct radiance)
  function updateLights() {
    emitterLightsFlat.length = 0;
    emitterColorsFlat.length = 0;
    const ents = query(world, [LightEmitter, LocalTransform]);
    let n = 0;
    for (let i = 0; i < ents.length && n < 8; i++) {
      const id = ents[i];
      const t = getMatrixTranslation(LocalTransform.matrix.getBatch(id));
      const r = hasComponent(world, id, Height) ? Height.value[id] * 0.5 : 0.5;
      emitterLightsFlat.push(t[0], t[1], t[2] + r, r);
      const c = Color.getArray(id); // rgba; rgb = c[0..2]
      emitterColorsFlat.push(c[0], c[1], c[2], LightEmitter.intensity[id]);
      n++;
    }
    voxel.setLights(emitterLightsFlat, n, emitterColorsFlat);
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

  // Spin every perf-grid instance slowly about Z so voxelize re-runs on moving occupancy
  // (mirrors real dynamic usage; cost is the same whether they move or not).
  function animatePerf(now: number) {
    if (!perf.animate || perfEntities.length === 0) return;
    const t = now * 0.0003;
    for (let i = 0; i < perfEntities.length; i++) {
      setMatrixRotateZ(LocalTransform.matrix.getBatch(perfEntities[i]), t + i * 0.3);
    }
  }

  // Animate the perf2 scene: every layer shape orbits a small loop around its home (so occupancy
  // truly MOVES each frame, not just spins), and every emitter sweeps the XY plane while bobbing
  // up and down through the full layer stack — i.e. flying between the layers.
  function animatePerf2(now: number) {
    if (!perf.animate || perf2Emitters.length === 0) return;
    const t = now * 0.001;
    // Shapes: small orbital wobble around the spawn home + slow spin.
    for (let i = 0; i < perfEntities.length; i++) {
      const hx = perf2Homes[i * 3];
      const hy = perf2Homes[i * 3 + 1];
      const hz = perf2Homes[i * 3 + 2];
      const ph = i * 0.7;
      const m = LocalTransform.matrix.getBatch(perfEntities[i]);
      setMatrixTranslate(
        m,
        hx + Math.cos(t * 0.8 + ph) * 0.9,
        hy + Math.sin(t * 0.9 + ph) * 0.9,
        hz + Math.sin(t * 0.6 + ph) * 0.5,
      );
      setMatrixRotateZ(m, t * 0.5 + ph);
    }
    // Emitters: wide XY sweep + Z bobbing across the whole stack (z0 .. top layer).
    const zMid = PERF2.z0 + ((PERF2.layers - 1) * PERF2.dz) * 0.5;
    const zAmp = ((PERF2.layers - 1) * PERF2.dz) * 0.5;
    for (let e = 0; e < perf2Emitters.length; e++) {
      const ph = e * 1.3;
      setMatrixTranslate(
        LocalTransform.matrix.getBatch(perf2Emitters[e]),
        Math.cos(t * 0.5 + ph) * (PERF2.spanX * 0.42),
        Math.sin(t * 0.42 + ph * 1.7) * (PERF2.spanY * 0.42),
        zMid + Math.sin(t * 0.7 + ph) * zAmp,
      );
    }
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
    if (SCENE === "perf2") animatePerf2(now);
    else animatePerf(now);
    // Auto-discover emitters (positions are now current) → cone importance-sampling lights.
    updateLights();
    execTransformSystem();
    shapeSystem.prepare();

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
      if (perf.draw) frameTick(encoder, delta);
      if (perf.sunDepth) voxel.sunDepth(encoder); // sun-POV depth feeding voxelize (A) + composite
      if (perf.voxelize) voxel.voxelize(encoder);
      if (perf.mips) voxel.mips(encoder);
      if (perf.probe) voxel.probe(encoder);
      if (perf.cone) voxel.cone(encoder);
      if (perf.composite) voxel.composite(encoder);
      present(encoder, voxel.compositeOutputTexture);
    } else {
      // Final lit image. Order: SDF G-buffer draw → (sun depth, only when the directional sun is
      // on) → voxelize → mips → cone gather → composite. The sun-POV depth map feeds the
      // composite's crisp cast shadow (and voxelize's shadowed sun injection).
      frameTick(encoder, delta);
      if (SunLight.enabled) {
        voxel.sunDepth(encoder);
      }
      voxel.voxelize(encoder);
      voxel.mips(encoder);
      voxel.probe(encoder);
      voxel.cone(encoder);
      voxel.composite(encoder);
      present(encoder, voxel.compositeOutputTexture);
    }

    // GPU time of this frame's submitted work — resolves when the GPU finishes, BEFORE the
    // vsync present, so it is not capped at 16.6 ms the way the rAF fps is. EMA-smoothed;
    // read the DELTA when a perf toggle flips to attribute cost to that pass.
    const tSubmit = performance.now();
    device.queue.submit([encoder.finish()]);
    // Serialize: wait for THIS frame's GPU work to fully finish before timing + encoding the
    // next. Removes the cross-frame queue backlog, so gpuMs is a clean single-frame number and
    // a pass toggle changes it unambiguously (diagnostic mode — not how a shipping loop runs).
    await device.queue.onSubmittedWorkDone();
    const dt = performance.now() - tSubmit;
    gpuMsEMA = gpuMsEMA ? gpuMsEMA * 0.8 + dt * 0.2 : dt;
    gpuMsMax = Math.max(gpuMsMax, gpuMsEMA);
    gpuPanel.update(gpuMsEMA, gpuMsMax);

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
