// engine demo — Rapier 3D physics driving the renderer (2.5D SDF) renderer.
//
// Gravity drops boxes and spheres onto a ground plane, lit by the VCT sun, viewed
// by a tilted orthographic camera. Z-up world; gravity (0,0,-9.81). A lil-gui panel
// spawns objects on demand. NOTE (2.5D renderer): translation is fully 3D, but only
// yaw + spheres render their rotation faithfully — a tumbling box looks upright.

import GUI from "lil-gui";
import { removeEntity } from "bitecs";
import { createEngine } from "./createEngine.ts";
import { getEngineComponents } from "./ECS/createEngineWorld.ts";
import { createGround, createRigidBox, createRigidSphere } from "./ECS/Entities/RigidShapes.ts";
import type { EngineWorld } from "./ECS/createEngineWorld.ts";
import type { TColor } from "../../renderer/src/ECS/Components/Common.ts";
import { SunLight } from "../../renderer/src/ECS/Systems/SunLight.ts";
import { RenderDI, type VoxelSystem } from "./DI/RenderDI.ts";
import {
  cameraAzimuth,
  cameraElevation,
  cameraZoom,
  setCameraAzimuth,
  setCameraElevation,
  setCameraPosition,
} from "../../renderer/src/ECS/Systems/ResizeSystem.ts";

// ── helpers ─────────────────────────────────────────────────────────────────

function hexToRgba(hex: string): TColor {
  const n = parseInt(hex.replace("#", ""), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, 1];
}

// Pleasant random color: random hue, fixed saturation/value.
function randomColor(): TColor {
  const h = Math.random();
  const s = 0.55;
  const v = 0.9;
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  const [r, g, b] = [
    [v, t, p],
    [q, v, p],
    [p, v, t],
    [p, q, v],
    [t, p, v],
    [v, p, q],
  ][i % 6];
  return [r, g, b, 1];
}

// ── scene + spawning ─────────────────────────────────────────────────────────

type Spawned = { eid: number };

function setupScene(world: EngineWorld): Spawned[] {
  const spawned: Spawned[] = [];

  // Ground: a wide, thin, fixed slab with its top face at z = 0.
  createGround(world, { size: 80, thickness: 1, z: 0, color: [0.18, 0.2, 0.24, 1] });

  // Sun (VCT): a soft warm key from above.
  SunLight.enabled = true;
  SunLight.angle = 2.4;
  SunLight.elevation = 0.95;
  SunLight.intensity = 0.9;
  SunLight.color = [1.0, 0.93, 0.82];

  // Camera: tilted top-down, framed on the origin.
  setCameraPosition(0, 0);
  cameraZoom.value = 14;
  cameraElevation.value = 60;
  cameraAzimuth.value = 45;

  return spawned;
}

function buildGui(world: EngineWorld, spawned: Spawned[]): GUI {
  const { LightEmitter } = getEngineComponents(world);

  const params = {
    shape: "box" as "box" | "sphere",
    size: 2, // box edge / sphere diameter (world units)
    height: 16, // drop height (z)
    spread: 6, // XY random spread radius around origin
    randomColor: true,
    color: "#cc7744",
    lightEmitter: false, // spawn as a light source (color = shape color)
    emitterIntensity: 6, // omni light intensity when lightEmitter is on
    count: 0, // live readout of dynamic body count
  };

  function spawnOne(): void {
    const x = (Math.random() - 0.5) * 2 * params.spread;
    const y = (Math.random() - 0.5) * 2 * params.spread;
    const z = params.height;
    const color = params.randomColor ? randomColor() : hexToRgba(params.color);
    const eid =
      params.shape === "sphere"
        ? createRigidSphere(world, { x, y, z, radius: params.size / 2, color })
        : createRigidBox(world, {
            x,
            y,
            z,
            sx: params.size,
            sy: params.size,
            sz: params.size,
            color,
          });
    if (params.lightEmitter) {
      // radius lifts the light center to the shape center (translation is the bottom).
      // Light color is the entity's Color; the voxel feed caps at 8 lights.
      LightEmitter.addComponent(world, eid, params.emitterIntensity, params.size / 2);
    }
    spawned.push({ eid });
    params.count = spawned.length;
  }

  function spawnMany(n: number): void {
    for (let i = 0; i < n; i++) spawnOne();
  }

  function clearDynamic(): void {
    // Despawn is pure ECS: removeEntity on the producer (main) world fires RigidBodyState's
    // onRemove → emits DESPAWN_BODY for the worker (fire-and-forget; eids never recycle, so
    // a late worker pose-write into the dead row is harmless). No method call here.
    for (let i = 0; i < spawned.length; i++) removeEntity(world, spawned[i].eid);
    spawned.length = 0;
    params.count = 0;
  }

  const gui = new GUI({ title: "Engine" });

  const spawn = gui.addFolder("Spawn");
  spawn.add(params, "shape", ["box", "sphere"]).name("shape");
  spawn.add(params, "size", 0.5, 6, 0.25).name("size");
  spawn.add(params, "height", 2, 30, 0.5).name("drop height");
  spawn.add(params, "spread", 0, 20, 0.5).name("xy spread");
  spawn.add(params, "randomColor").name("random color");
  spawn.addColor(params, "color").name("color");
  spawn.add(params, "lightEmitter").name("light emitter");
  spawn.add(params, "emitterIntensity", 0, 30, 0.5).name("emitter intensity");
  spawn.add({ spawn1: () => spawnOne() }, "spawn1").name("Spawn ×1");
  spawn.add({ spawn10: () => spawnMany(10) }, "spawn10").name("Spawn ×10");
  spawn.add({ clear: () => clearDynamic() }, "clear").name("Clear");
  spawn.add(params, "count").name("dynamic bodies").disable().listen();

  // seed a handful so the scene isn't empty on load
  spawnMany(6);

  return gui;
}

// ── VCT tuning GUI ─────────────────────────────────────────────────────────────
// Mirrors renderer/src/demo.ts's voxel folders, wired to the SAME voxel system
// (RenderDI.voxel). Baked-config controls recompile the GI shaders on RELEASE
// (.onFinishChange → voxel.rebuild()), not per drag tick; the sun + cone-resolution
// are read live. So both packages share one render config — tune here, the values
// live in voxel.config (defaults come from voxelConfig.ts, common to both demos).
function addVoxelControls(gui: GUI, voxel: VoxelSystem): void {
  const rebuild = () => voxel.rebuild();

  // Graininess: voxel size in world units. Rebuilds the 3D textures on release.
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

  // Sun (directional key, read live by voxelize + composite).
  const sun = gui.addFolder("Sun");
  sun.add(SunLight, "enabled").name("sun enabled");
  sun.add(SunLight, "angle", 0, Math.PI * 2, 0.01).name("sun angle");
  sun.add(SunLight, "elevation", 0, Math.PI / 2, 0.01).name("sun elevation");
  sun.add(SunLight, "intensity", 0, 5, 0.05).name("sun intensity");
  sun.addColor(SunLight, "color", 1).name("sun color"); // rgbScale=1 → 0..1 floats

  // Cone GI: per-pixel AIMED emitter cones (sharp direct + shadow). Baked → rebuild on release.
  const coneFolder = gui.addFolder("Cone GI");
  coneFolder
    .add(voxel.config, "emitterDirect", 0, 8, 0.1)
    .name("emitter direct strength")
    .onFinishChange(rebuild);
  coneFolder
    .add(voxel.config, "emitterFalloff", 0, 4, 0.05)
    .name("emitter falloff")
    .onFinishChange(rebuild);
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
  const coneResCfg = { scale: voxel.coneScale };
  coneFolder
    .add(coneResCfg, "scale", { "half-res (2)": 2, "quarter-res (4)": 4, "eighth-res (8)": 8 })
    .name("cone resolution")
    .onChange((s: number) => voxel.setConeScale(s));
  coneFolder.add(voxel.config, "aimedSteps", 8, 64, 1).name("aimed steps").onFinishChange(rebuild);
  coneFolder
    .add(voxel.config, "aimedAlphaCut", 0.5, 1, 0.01)
    .name("aimed alpha cut")
    .onFinishChange(rebuild);

  // Probe GI: the low-res irradiance-probe volume (fill/bounce) + short contact-AO cones.
  const probeFolder = gui.addFolder("Probe GI");
  probeFolder
    .add(voxel.config, "conesPerProbe", [8, 16, 32, 64, 128])
    .name("cones / probe")
    .onFinishChange(rebuild);
  probeFolder
    .add(voxel.config, "aoConeCount", 0, 8, 1)
    .name("AO cones (contact)")
    .onFinishChange(rebuild);
  probeFolder.add(voxel.config, "aoReach", 1, 16, 0.5).name("AO reach").onFinishChange(rebuild);
  // Probe-volume blur radius: 3D Gaussian smoothing of the SH bounce so a moving source's fill
  // stops stepping by probe cells. 0 = off. Cheap (O(probes·kernel), not ·cones).
  probeFolder
    .add(voxel.config, "probeBlurRadius", 0, 4, 1)
    .name("probe blur radius")
    .onFinishChange(rebuild);

  // Composite: ambient floor + HDR exposure + sun-dim penumbra.
  const compositeFolder = gui.addFolder("Composite");
  compositeFolder
    .add(voxel.config, "ambient", 0, 0.5, 0.01)
    .name("composite ambient")
    .onFinishChange(rebuild);
  compositeFolder
    .add(voxel.config, "exposure", 0.1, 4, 0.05)
    .name("exposure")
    .onFinishChange(rebuild);
  compositeFolder
    .add(voxel.config, "penumbra", 0, 12, 0.5)
    .name("penumbra (sun-dim)")
    .onFinishChange(rebuild);
  // Base sun-shadow PCF radius applied even at full sun → smooths the shadow-map texel staircase.
  // 1 = near-hard. The sun frustum also auto-fits the camera view, so steps shrink on zoom.
  compositeFolder
    .add(voxel.config, "shadowBaseSpread", 1, 6, 0.25)
    .name("shadow softness")
    .onFinishChange(rebuild);
}

// ── main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const engine = await createEngine({ canvas });

  const spawned = setupScene(engine.world);
  const gui = buildGui(engine.world, spawned);

  const orbit = { enabled: true };
  gui.add(orbit, "enabled").name("orbit camera");

  // VCT tuning folders — wired to the SAME voxel system the renderer demo tunes
  // (RenderDI.voxel, set by createRenderTarget). createEngine({canvas}) has already
  // built the render target, so the handle is live here.
  if (RenderDI.voxel) addVoxelControls(gui, RenderDI.voxel);

  let then = performance.now();
  function loop(now: number): void {
    const delta = Math.min(now - then, 16.6667) / 1000; // clamp, seconds
    then = now;
    if (orbit.enabled) setCameraAzimuth(cameraAzimuth.value + delta * 10);
    // Re-clamp elevation in case a drag pushed it out of range.
    setCameraElevation(cameraElevation.value);
    engine.tick(delta);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f88;padding:20px">${(err as Error)?.stack ?? err}</pre>`;
  console.error(err);
});
