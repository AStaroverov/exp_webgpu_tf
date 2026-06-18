/**
 * Standalone 3D renderer test scene — does NOT use the engine (no physics, no
 * gameplay, no createGame). It drives `packages/renderer` directly to validate
 * the Phase 0/1 migration:
 *   - perspective reverse-Z view-projection + orbital camera (ResizeSystem)
 *   - cube-impostor SDF kernel for Box3D / Sphere3D (sdf.shader.ts)
 *   - MRT G-buffer (albedo + world normal + emission) -> stopgap Lambert composite
 *
 * Drag = orbit, wheel = dolly. Spinning shapes make the lighting/normals obvious.
 */
import { addEntity } from "bitecs";
import { mat4 } from "gl-matrix";

import { createWorld, getRenderComponents } from "../../../renderer/src/ECS/world.ts";
import { addTransformComponents } from "../../../renderer/src/ECS/Components/Transform.ts";
import { ShapeKind } from "../../../renderer/src/ECS/Components/Shape.ts";
import { initWebGPU } from "../../../renderer/src/gpu.ts";
import { createFrameTextures, createFrameTick } from "../../../renderer/src/WGSL/createFrame.ts";
import { createDrawShapeSystem } from "../../../renderer/src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts";
import { createCompositeStopgap } from "../../../renderer/src/WGSL/createCompositeStopgap.ts";
import { createRadianceCascades } from "../../../renderer/src/WGSL/createRadianceCascades.ts";
import { createPresent } from "../../../renderer/src/WGSL/createPresent.ts";
import { orbit, orbitBy, setCameraZoom } from "../../../renderer/src/ECS/Systems/ResizeSystem.ts";
import GUI from "lil-gui";

type Color = [number, number, number, number];

// World is +Y up (ResizeSystem lookAt up = (0,1,0)); the ground is the X–Z plane
// at y=0, objects rise along +Y. Camera orbits the origin.
// Horizontal-circle flight path (for light emitters), animated each frame.
interface Flight {
  cx: number;
  cz: number;
  radius: number;
  height: number;
  speed: number;
  phase: number;
  bob: number; // vertical wobble amplitude
}
interface SceneObject {
  id: number;
  pos: [number, number, number];
  spin: number; // rad/sec around +Y (0 = static)
  flight?: Flight; // if set, pos is driven by the flight path each frame
}

const objects: SceneObject[] = [];

function spawn(
  world: ReturnType<typeof createWorld>,
  kind: ShapeKind,
  pos: [number, number, number],
  values: number[],
  color: Color,
  opts: { roundness?: number; spin?: number; emitter?: number; flight?: Flight } = {},
) {
  const { Shape, Color, Roundness, LightEmitter } = getRenderComponents(world);
  const id = addEntity(world);
  addTransformComponents(world, id);

  Shape.addComponent(
    world,
    id,
    kind,
    values[0] ?? 0,
    values[1] ?? 0,
    values[2] ?? 0,
    values[3] ?? 0,
    values[4] ?? 0,
    values[5] ?? 0,
  );
  Color.addComponent(world, id, color[0], color[1], color[2], color[3]);
  Roundness.addComponent(world, id, opts.roundness ?? 0);
  // Emitter: emission = color * intensity (HDR) lands in the G-buffer's emission
  // channel; the RC march seeds radiance from it -> colored GI bleed.
  if (opts.emitter !== undefined) LightEmitter.addComponent(world, id, opts.emitter);

  objects.push({ id, pos, spin: opts.spin ?? 0, flight: opts.flight });
  return id;
}

const PALETTE: Color[] = [
  [0.85, 0.25, 0.25, 1],
  [0.9, 0.6, 0.2, 1],
  [0.9, 0.85, 0.25, 1],
  [0.3, 0.75, 0.35, 1],
  [0.25, 0.55, 0.9, 1],
  [0.6, 0.35, 0.85, 1],
];

function buildScene(world: ReturnType<typeof createWorld>) {
  // Ground: a large flat box (reliable volume for the impostor to trace).
  spawn(world, ShapeKind.Box3D, [0, -4, 0], [220, 4, 220], [0.78, 0.66, 0.5, 1]);

  const xs = [-150, -90, -30, 30, 90, 150];

  // Row of spheres (radius 26), sitting on the ground.
  xs.forEach((x, i) => {
    spawn(world, ShapeKind.Sphere3D, [x, 26, -70], [26], PALETTE[i % PALETTE.length]);
  });

  // Row of spinning ROUNDED boxes (half-extents 24, corner radius 4) — rotation
  // takes faces near edge-on, stressing the IQ-style rounded-box sphere-trace.
  xs.forEach((x, i) => {
    spawn(world, ShapeKind.Box3D, [x, 24, 70], [24, 24, 24], PALETTE[(i + 3) % PALETTE.length], {
      roundness: 4,
      spin: 0.6 + i * 0.12,
    });
  });

  // A tall pillar + a big center sphere to read depth/occlusion.
  spawn(world, ShapeKind.Box3D, [0, 60, 0], [14, 60, 14], [0.7, 0.7, 0.75, 1], { spin: 0.3 });
  spawn(world, ShapeKind.Sphere3D, [-90, 45, 0], [40], [0.95, 0.95, 0.97, 1]);

  // A thin flat tile (a "plane" is just a thin Box3D — analytic ray-box).
  spawn(world, ShapeKind.Box3D, [170, 8, 0], [40, 1, 60], [0.4, 0.5, 0.6, 1]);

  // Flying light emitters: small bright spheres on circular paths. Their emission
  // (color * intensity, HDR) seeds the Radiance Cascades -> colored GI bleeds onto
  // the surfaces below as they pass over. This is what makes RC more than AO.
  const lights: { color: Color; phase: number; radius: number; height: number }[] = [
    { color: [1.0, 0.25, 0.2, 1], phase: 0.0, radius: 150, height: 55 },
    { color: [0.25, 1.0, 0.35, 1], phase: 2.1, radius: 110, height: 80 },
    { color: [0.3, 0.5, 1.0, 1], phase: 4.2, radius: 180, height: 45 },
  ];
  for (const l of lights) {
    spawn(world, ShapeKind.Sphere3D, [0, l.height, 0], [10], l.color, {
      emitter: 5, // HDR intensity
      flight: {
        cx: 0,
        cz: 0,
        radius: l.radius,
        height: l.height,
        speed: 0.5,
        phase: l.phase,
        bob: 18,
      },
    });
  }
}

function setupOrbitInput(canvas: HTMLCanvasElement) {
  orbit.target[0] = 0;
  orbit.target[1] = 20;
  orbit.target[2] = 0;
  orbit.yaw = 0.6;
  orbit.pitch = 0.5; // positive = eye above the ground, looking down
  orbit.distance = 480;

  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", (e) => {
    dragging = false;
    canvas.releasePointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    orbitBy(-dx * 0.005, -dy * 0.005);
  });
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      // wheel up (negative deltaY) -> zoom in. setCameraZoom maps zoom->distance.
      const factor = Math.exp(-e.deltaY * 0.001);
      const baseDistance = 600; // ResizeSystem BASE_DISTANCE
      setCameraZoom((baseDistance / orbit.distance) * factor);
    },
    { passive: false },
  );
}

async function main() {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const side = Math.min(window.innerWidth, window.innerHeight) * window.devicePixelRatio;
  canvas.width = side;
  canvas.height = side;

  const world = createWorld();
  buildScene(world);
  setupOrbitInput(canvas);

  const { device, context } = await initWebGPU(canvas);
  const textures = createFrameTextures(device, canvas);
  const shapeSystem = createDrawShapeSystem({ world, device });

  const frameTick = createFrameTick(
    {
      ...textures,
      canvas,
      device,
      background: [226, 192, 146, 255].map((v) => v / 255) as unknown as GPUColor,
      getPixelRatio: () => window.devicePixelRatio,
    },
    ({ passEncoder }) => {
      shapeSystem.prepare();
      shapeSystem.drawShapes(passEncoder);
    },
  );

  const composite = createCompositeStopgap(device);

  // RC cascade textures are managed HERE (not via createFrameTextures) so their
  // resolution is live-tunable: rcDownscale changes the texture size, which means
  // recreating the textures + the RC factory (not a uniform). litTexture (the
  // gather output) stays full-res regardless.
  const RC_CFG = { downscale: 1.0 };
  function makeCascPair(downscale: number): [GPUTexture, GPUTexture] {
    const w = Math.max(1, Math.floor(canvas.width * downscale));
    const h = Math.max(1, Math.floor(canvas.height * downscale));
    const mk = () =>
      device.createTexture({
        size: [w, h, 1],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
    return [mk(), mk()];
  }
  let [cascA, cascB] = makeCascPair(RC_CFG.downscale);
  let radianceCascades = createRadianceCascades(device, { cascA, cascB });
  const rcParams = radianceCascades.params; // stable ref reused across rebuilds
  function rebuildRC(downscale: number) {
    cascA.destroy();
    cascB.destroy();
    [cascA, cascB] = makeCascPair(downscale);
    radianceCascades = createRadianceCascades(device, { cascA, cascB }, rcParams);
  }

  // keep `composite` referenced for the 1-line stopgap A/B toggle below.
  void composite;
  const present = createPresent(device, context);

  // --- lil-gui: live RC tuning. params mutate in place, applied next frame. ---
  const gui = new GUI({ title: "Radiance Cascades" });
  const p = rcParams;
  gui
    .add(RC_CFG, "downscale", 0.25, 1.0, 0.05)
    .name("RC resolution")
    .onFinishChange(rebuildRC); // realloc on release, not mid-drag
  // Geometry: rounded-box sphere-trace budget (dominant zoom-in cost).
  gui.add(shapeSystem.params, "roundSteps", 4, 96, 1).name("rounded-box steps");
  gui.add(p, "marchSteps", 4, 96, 1).name("march steps");
  gui.add(p, "zBias", 0, 40, 0.25).name("occlusion bias (world)");
  gui.add(p, "baseIntervalPx", 1, 64, 1).name("base interval px");
  gui.add(p, "ambient", 0, 1, 0.01).name("ambient floor");
  gui.add(p, "ambientFill", 0, 1, 0.01).name("hit albedo fill");
  gui.addColor(p, "sky", 1).name("sky / miss color");
  // Edge-aware denoise of the irradiance (0 radius = off).
  gui.add(p, "denoiseRadius", 0, 3, 1).name("denoise radius");
  gui.add(p, "denoiseWorldSigma", 1, 60, 1).name("denoise world σ");
  gui.add(p, "denoiseNormalPow", 1, 128, 1).name("denoise normal pow");

  const { GlobalTransform, LocalTransform } = getRenderComponents(world);
  const m = mat4.create();

  let prev = performance.now();
  const loop = (now: number) => {
    const delta = Math.min(33.3, now - prev);
    prev = now;
    const t = now / 1000;

    // Animate transforms (write straight into GlobalTransform — no hierarchy,
    // so no transform system is needed; keep LocalTransform in sync for parity).
    for (let i = 0; i < objects.length; i++) {
      const o = objects[i];
      if (o.flight) {
        const fl = o.flight;
        o.pos[0] = fl.cx + fl.radius * Math.cos(t * fl.speed + fl.phase);
        o.pos[2] = fl.cz + fl.radius * Math.sin(t * fl.speed + fl.phase);
        o.pos[1] = fl.height + fl.bob * Math.sin(t * fl.speed * 1.7 + fl.phase);
      }
      mat4.fromTranslation(m, o.pos);
      if (o.spin !== 0) mat4.rotateY(m, m, t * o.spin);
      GlobalTransform.matrix.setBatch(o.id, m);
      LocalTransform.matrix.setBatch(o.id, m);
    }

    const encoder = device.createCommandEncoder();
    frameTick(encoder, delta); // scene -> G-buffer
    // --- RC lighting (Phase 2). Toggle: comment this + present(litTexture) and
    //     uncomment the stopgap composite()/present() below for an A/B. ---
    radianceCascades(
      encoder,
      textures.gAlbedo,
      textures.gNormal,
      textures.gEmission,
      textures.depthTexture,
      textures.litTexture,
    ); // G-buffer -> radiance-cascade-lit
    present(encoder, textures.litTexture); // -> swapchain
    // composite(encoder, textures.gAlbedo, textures.gNormal, textures.gEmission, textures.compositeTexture); present(encoder, textures.compositeTexture); // <-- stopgap fallback (1-line toggle)
    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

main().catch((err) => {
  console.error(err);
  document.body.innerHTML = `<pre style="color:#f55;padding:16px;white-space:pre-wrap">${String(
    err?.stack ?? err,
  )}</pre>`;
});
