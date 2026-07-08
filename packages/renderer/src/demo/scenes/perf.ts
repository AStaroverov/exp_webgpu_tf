import type GUI from "lil-gui";
import { getRenderComponents, type RenderWorld } from "../../ECS/world.ts";
import { SunLight } from "../../ECS/Systems/SunLight.ts";
import { setMatrixRotateZ } from "../../ECS/Components/Transform.ts";
import { createCircle, createRectangle, createSphere } from "../../ECS/Entities/Shapes.ts";
import type { DemoScene } from "./types.ts";
import { addPerfFolder, perfToggles } from "./perfToggles.ts";

// "perf" — a dense grid of mixed shapes to stress voxelize (O(voxels×instances)) and the SDF
// draw (per-fragment march + overdraw), plus a few emitters. Heavy enough to push GPU time well
// above the inspector's ~4 ms floor so the per-pass toggles read clearly.
export function createPerfScene(world: RenderWorld): DemoScene {
  const { LocalTransform, LightEmitter } = getRenderComponents(world);
  const perfEntities: number[] = []; // grid instance ids, spun by animate

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

  return {
    // Spin every grid instance slowly about Z so voxelize re-runs on moving occupancy
    // (mirrors real dynamic usage; cost is the same whether they move or not).
    animate(now: number) {
      if (!perfToggles.animate) return;
      const t = now * 0.0003;
      for (let i = 0; i < perfEntities.length; i++) {
        setMatrixRotateZ(LocalTransform.matrix.getBatch(perfEntities[i]), t + i * 0.3);
      }
    },
    setupGUI(gui: GUI) {
      addPerfFolder(gui);
    },
  };
}
