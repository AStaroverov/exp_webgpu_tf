/**
 * TEMPORARY diagnostic — counts renderable SDF-shape entities and the main
 * dynamic contributors once per second, to find what overflows the 10k instance
 * buffer (MAX_INSTANCE_COUNT in sdf.shader.ts). DELETE this file + its wiring in
 * createGame.ts once the leak is identified.
 */

import { query } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";

export function createShapeCountDiagnosticSystem({ world } = GameDI) {
  const {
    Shape,
    Color,
    GlobalTransform,
    Bullet,
    TreadMark,
    DestroyByTimeout,
    VehiclePart,
    VFX,
    Wander,
    Destroy,
  } = getGameComponents(world);
  let acc = 0;
  let prevSdf = 0;

  return function diagnostic(delta: number) {
    acc += delta;
    if (acc < 1000) return;
    acc = 0;

    const sdf = query(world, [GlobalTransform, Shape, Color]).length;
    const bullets = query(world, [Bullet]).length;
    const tread = query(world, [TreadMark]).length;
    const timed = query(world, [DestroyByTimeout]).length;
    const parts = query(world, [VehiclePart]).length;
    const particles = query(world, [Wander]).length; // stream (flame/frost) particles
    const vfx = query(world, [VFX]).length;
    const pending = query(world, [Destroy]).length;
    const accountedFor = parts + tread + bullets + particles;
    const other = sdf - accountedFor;
    const growth = sdf - prevSdf;
    prevSdf = sdf;

    if (sdf > 8000) {
      // eslint-disable-next-line no-console
      console.warn(
        `[shape-count] SDF=${sdf} (${growth >= 0 ? "+" : ""}${growth}/s) cap=10000 | ` +
          `parts=${parts} treadMarks=${tread} particles=${particles} bullets=${bullets} ` +
          `vfx=${vfx} other≈${other} | destroyByTimeout=${timed} pendingDestroy=${pending}`,
      );
    }
  };
}
