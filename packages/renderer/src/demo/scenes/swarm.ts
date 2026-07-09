import type GUI from "lil-gui";
import { getRenderComponents, type RenderWorld } from "../../ECS/world.ts";
import { SunLight } from "../../ECS/Systems/SunLight.ts";
import { applyMatrixRotateZ, setMatrixTranslate } from "../../ECS/Components/Transform.ts";
import { createCircle, createRectangle, createSphere } from "../../ECS/Entities/Shapes.ts";
import type { DemoScene } from "./types.ts";

// "swarm" — MANY small emitters hovering over the floor + sparse large occluders: the
// many-lights harness for the uncapped emitter list. Exercises the clustered cull (lights
// spread over many cells) AND the round-robin subsample (crank `count` above aimedPerFrame ×
// cells). Deterministic layout (seeded LCG) so reloads and A/B config flips compare the same
// scene.
export function createSwarmScene(world: RenderWorld): DemoScene {
  const { LocalTransform, LightEmitter } = getRenderComponents(world);

  const SWARM = { count: 64, span: 46, zMin: 0.8, zMax: 2.4 };
  const swarmCfg = { animate: true, speed: 1.0 };
  const swarmEmitters: number[] = []; // emitter ids, parallel to swarmHomes
  const swarmHomes: number[] = []; // x,y,z hover anchor per emitter

  // Dim warm sun: enough to shape the large objects, dim enough that the emitter light (and
  // its subsample/cluster artifacts, if any) dominates the read.
  SunLight.enabled = true;
  SunLight.angle = 2.4;
  SunLight.elevation = 0.95;
  SunLight.intensity = 0.35;
  SunLight.color = [1.0, 0.93, 0.82];

  // Ground slab — the canvas the swarm light pools paint on.
  createRectangle(world, {
    x: 0,
    y: 0,
    z: -0.4,
    width: 64,
    height: 64,
    depth: 0.4,
    color: [0.2, 0.2, 0.22, 1],
  });

  // Sparse LARGE occluders: tall enough to cut through the hover band (zMin..zMax) so nearby
  // emitters cast readable soft shadows / occlusion onto the floor and each other.
  createRectangle(world, {
    x: -14,
    y: 10,
    z: 0,
    width: 5,
    height: 5,
    depth: 7,
    color: [0.75, 0.55, 0.4, 1],
  });
  createRectangle(world, {
    x: 12,
    y: -12,
    z: 0,
    width: 8,
    height: 2.5,
    depth: 5,
    color: [0.7, 0.7, 0.72, 1],
  });
  const swarmSlab = createRectangle(world, {
    x: 2,
    y: 14,
    z: 0,
    width: 10,
    height: 1.6,
    depth: 4.5,
    color: [0.6, 0.62, 0.7, 1],
  });
  applyMatrixRotateZ(LocalTransform.matrix.getBatch(swarmSlab), 0.5);
  createSphere(world, { x: -12, y: -10, z: 0, radius: 3.2, color: [0.9, 0.9, 0.92, 1] });
  createSphere(world, { x: 15, y: 8, z: 0, radius: 2.4, color: [0.85, 0.5, 0.45, 1] });
  createCircle(world, { x: 0, y: -14, z: 0, radius: 2.2, height: 5, color: [0.5, 0.7, 0.6, 1] });
  createCircle(world, { x: -2, y: 2, z: 0, radius: 1.6, height: 6, color: [0.55, 0.55, 0.85, 1] });

  // The swarm: SWARM.count small emitter spheres hovering in the zMin..zMax band. Seeded LCG →
  // identical layout every reload. Positions rejected inside a small keep-out around each large
  // occluder are NOT needed — an emitter drifting into geometry is itself a useful stress case.
  let seed = 1234567;
  const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
  // Hue palette cycled with per-light jitter — distinct pools without full-random mud.
  const swarmPalette: [number, number, number][] = [
    [1.0, 0.55, 0.2],
    [0.3, 0.6, 1.0],
    [0.6, 1.0, 0.35],
    [1.0, 0.35, 0.7],
    [0.95, 0.9, 0.55],
    [0.4, 0.95, 0.85],
  ];
  for (let i = 0; i < SWARM.count; i++) {
    const x = (rnd() - 0.5) * SWARM.span;
    const y = (rnd() - 0.5) * SWARM.span;
    const z = SWARM.zMin + rnd() * (SWARM.zMax - SWARM.zMin);
    const tint = swarmPalette[i % swarmPalette.length];
    const jitter = 0.85 + rnd() * 0.3;
    const id = createSphere(world, {
      x,
      y,
      z,
      radius: 0.25 + rnd() * 0.2,
      color: [tint[0] * jitter, tint[1] * jitter, tint[2] * jitter, 1],
    });
    LightEmitter.addComponent(world, id, 1.5 + rnd() * 1.5);
    swarmEmitters.push(id);
    swarmHomes.push(x, y, z);
  }

  return {
    // The swarm spreads wide — pull back to frame the field.
    cameraZoom: 11,
    // Hover: each emitter bobs on its own phase (small Z sine + a slow XY drift circle) around
    // its spawn anchor — light pools visibly float over the floor, positions change every frame
    // (re-binned by the clustered cull, reprojected by the temporal history).
    animate(now: number) {
      if (!swarmCfg.animate) return;
      const t = now * 0.001 * swarmCfg.speed;
      for (let i = 0; i < swarmEmitters.length; i++) {
        const ph = i * 0.917;
        setMatrixTranslate(
          LocalTransform.matrix.getBatch(swarmEmitters[i]),
          swarmHomes[i * 3] + Math.cos(t * 0.35 + ph) * 0.8,
          swarmHomes[i * 3 + 1] + Math.sin(t * 0.3 + ph * 1.6) * 0.8,
          swarmHomes[i * 3 + 2] + Math.sin(t * 0.9 + ph) * 0.35,
        );
      }
    },
    setupGUI(gui: GUI) {
      const sf = gui.addFolder("Swarm");
      sf.add(swarmCfg, "animate").name("hover animation");
      sf.add(swarmCfg, "speed", 0.1, 4, 0.1).name("hover speed");
      sf.open();
    },
  };
}
