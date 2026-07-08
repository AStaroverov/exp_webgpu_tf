import type GUI from "lil-gui";
import { getRenderComponents, type RenderWorld } from "../../ECS/world.ts";
import { SunLight } from "../../ECS/Systems/SunLight.ts";
import { setMatrixRotateZ, setMatrixTranslate } from "../../ECS/Components/Transform.ts";
import { createCircle, createRectangle, createSphere } from "../../ECS/Entities/Shapes.ts";
import type { DemoScene } from "./types.ts";

// "final" — static set + 4 dynamic objects, one per animated axis (position / angle / size /
// intensity), each mutated every frame so the change re-voxelizes and the GI updates live.
export function createFinalScene(world: RenderWorld): DemoScene {
  const { LocalTransform, LightEmitter, Shape } = getRenderComponents(world);

  const finalDyn = { animate: true, speed: 1.0 };

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
  const dynOrbitEmitter = createSphere(world, {
    x: 9,
    y: 0,
    z: 3,
    radius: 3,
    color: [1.0, 0.55, 0.2, 1],
  });
  LightEmitter.addComponent(world, dynOrbitEmitter, 10.0);
  // (2) ANGLE — a tall slab rotating about Z: a moving occluder → shifting AO/bounce.
  const dynRotBox = createRectangle(world, {
    x: 0,
    y: 3,
    z: 0,
    width: 5,
    height: 1.4,
    depth: 6,
    color: [0.8, 0.45, 0.5, 1],
  });
  // (3) SIZE — a solid sphere whose radius pulses (re-voxelized each frame).
  const dynSizeSphere = createSphere(world, {
    x: -9,
    y: -6,
    z: 0,
    radius: 2.0,
    color: [0.45, 0.6, 0.85, 1],
  });
  // (4) INTENSITY — a cool emitter pulsing its emission.
  const dynPulseEmitter = createSphere(world, {
    x: -9,
    y: 2,
    z: 2.2,
    radius: 0.9,
    color: [0.35, 0.6, 1.0, 1],
  });
  LightEmitter.addComponent(world, dynPulseEmitter, 2.5);

  return {
    animate(now: number) {
      if (!finalDyn.animate) return;
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
      // (3) size — radius 1..3, re-voxelized via the Shape setter (the sphere radius is its
      // own full Z extent; center-origin keeps it about its transform center).
      const r = 2.0 + 1.0 * Math.sin(t * 1.2);
      Shape.setSphere$(dynSizeSphere, r);
      // (4) intensity — emission 0.2..3.8 (kept positive; negative would mean "directional").
      LightEmitter.set$(dynPulseEmitter, 2.0 + 1.8 * Math.sin(t * 1.6), 0);
    },
    setupGUI(gui: GUI) {
      const f = gui.addFolder("Final scene");
      f.add(finalDyn, "animate").name("animate");
      f.add(finalDyn, "speed", 0, 3, 0.05).name("speed");
    },
  };
}
