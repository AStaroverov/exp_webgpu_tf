import type GUI from "lil-gui";
import { getRenderComponents, type RenderWorld } from "../../ECS/world.ts";
import { SunLight } from "../../ECS/Systems/SunLight.ts";
import { setMatrixRotateZ, setMatrixTranslate } from "../../ECS/Components/Transform.ts";
import { createCircle, createRectangle, createSphere } from "../../ECS/Entities/Shapes.ts";
import type { DemoScene } from "./types.ts";
import { addPerfFolder, perfToggles } from "./perfToggles.ts";

// "perf2" — 5 stacked Z-layers of small MOVING shapes (150/layer = 750) with emitters weaving
// through the gaps. The target workload — many small dynamic objects re-voxelized every frame.
export function createPerf2Scene(world: RenderWorld): DemoScene {
  const { LocalTransform, LightEmitter } = getRenderComponents(world);

  // Geometry shared between the spawn loop and animate (homes drift around these).
  const PERF2 = { layers: 5, cols: 15, rows: 10, spanX: 42, spanY: 28, z0: 1.0, dz: 2.4 };
  const perfEntities: number[] = []; // layer shape ids
  const perf2Homes: number[] = []; // x,y,z per shape, parallel to perfEntities (orbit anchors)
  const perf2Emitters: number[] = []; // ids of the emitters flying between the layers

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

  // Emitters that fly through the gaps between layers (animated below). Distinct hues.
  // (The light count is uncapped now — the "swarm" scene is the dedicated many-lights harness.)
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
    LightEmitter.addComponent(world, id, 1.0);
    perf2Emitters.push(id);
  }

  return {
    // 5 layers stack tall over a wide span — pull back to frame the whole thing.
    cameraZoom: 9,
    // Every layer shape orbits a small loop around its home (so occupancy truly MOVES each
    // frame, not just spins), and every emitter sweeps the XY plane while bobbing up and down
    // through the full layer stack — i.e. flying between the layers.
    animate(now: number) {
      if (!perfToggles.animate) return;
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
      const zMid = PERF2.z0 + (PERF2.layers - 1) * PERF2.dz * 0.5;
      const zAmp = (PERF2.layers - 1) * PERF2.dz * 0.5;
      for (let e = 0; e < perf2Emitters.length; e++) {
        const ph = e * 1.3;
        setMatrixTranslate(
          LocalTransform.matrix.getBatch(perf2Emitters[e]),
          Math.cos(t * 0.5 + ph) * (PERF2.spanX * 0.42),
          Math.sin(t * 0.42 + ph * 1.7) * (PERF2.spanY * 0.42),
          zMid + Math.sin(t * 0.7 + ph) * zAmp,
        );
      }
    },
    setupGUI(gui: GUI) {
      addPerfFolder(gui);
    },
  };
}
