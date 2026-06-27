// engine demo — Rapier 3D physics driving the renderer3d_2 (2.5D SDF) renderer.
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
import type { PhysicalWorld } from "./Physics/initPhysicalWorld.ts";
import type { TColor } from "../../renderer3d_2/src/ECS/Components/Common.ts";
import { SunLight } from "../../renderer3d_2/src/ECS/Systems/SunLight.ts";
import {
  cameraAzimuth,
  cameraElevation,
  cameraZoom,
  setCameraAzimuth,
  setCameraElevation,
  setCameraPosition,
} from "../../renderer3d_2/src/ECS/Systems/ResizeSystem.ts";

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

type Spawned = { eid: number; pid: number };

function setupScene(world: EngineWorld, pw: PhysicalWorld): Spawned[] {
  const spawned: Spawned[] = [];

  // Ground: a wide, thin, fixed slab with its top face at z = 0.
  createGround(world, pw, { size: 80, thickness: 1, z: 0, color: [0.18, 0.2, 0.24, 1] });

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

function buildGui(world: EngineWorld, pw: PhysicalWorld, spawned: Spawned[]): GUI {
  const { RigidBodyRef, LightEmitter } = getEngineComponents(world);

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
    const [eid, pid] =
      params.shape === "sphere"
        ? createRigidSphere(world, pw, { x, y, z, radius: params.size / 2, color })
        : createRigidBox(world, pw, { x, y, z, sx: params.size, sy: params.size, sz: params.size, color });
    if (params.lightEmitter) {
      // radius lifts the light center to the shape center (translation is the bottom).
      // Light color is the entity's Color; the voxel feed caps at 8 lights.
      LightEmitter.addComponent(world, eid, params.emitterIntensity, params.size / 2);
    }
    spawned.push({ eid, pid });
    params.count = spawned.length;
  }

  function spawnMany(n: number): void {
    for (let i = 0; i < n; i++) spawnOne();
  }

  function clearDynamic(): void {
    for (let i = 0; i < spawned.length; i++) {
      const { eid, pid } = spawned[i];
      const body = pw.getRigidBody(pid);
      if (body) pw.removeRigidBody(body);
      RigidBodyRef.clear(eid);
      removeEntity(world, eid);
    }
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

  const scene = gui.addFolder("Scene").close();
  scene.add(SunLight, "enabled").name("sun enabled");
  scene.add(SunLight, "intensity", 0, 5, 0.05).name("sun intensity");

  // seed a handful so the scene isn't empty on load
  spawnMany(6);

  return gui;
}

// ── main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const canvas = document.getElementById("c") as HTMLCanvasElement;
  const engine = await createEngine({ canvas });

  const spawned = setupScene(engine.world, engine.physicalWorld);
  const gui = buildGui(engine.world, engine.physicalWorld, spawned);

  const orbit = { enabled: true };
  gui.add(orbit, "enabled").name("orbit camera");

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
