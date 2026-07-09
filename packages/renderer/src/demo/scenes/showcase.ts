import { mat4 } from "gl-matrix";
import { getRenderComponents, type RenderWorld } from "../../ECS/world.ts";
import { applyMatrixRotateZ } from "../../ECS/Components/Transform.ts";
import {
  createCircle,
  createParallelogram,
  createRectangle,
  createSphere,
  createTrapezoid,
  createTriangle,
} from "../../ECS/Entities/Shapes.ts";
import type { DemoScene } from "./types.ts";

// "showcase" — at least one of every shape kind, plus a stacked platform and several lights.
export function createShowcaseScene(world: RenderWorld): DemoScene {
  const { LocalTransform, LightEmitter } = getRenderComponents(world);

  // Ground slab (flat-ish wide rectangle at baseZ below 0).
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

  // Three non-symmetric boxes, each tilted about ONE axis (X / Y / Z) to prove the
  // impostor honors full 3D rotation, not just yaw.
  const tiltX = createRectangle(world, {
    x: 3,
    y: -7,
    z: 3,
    width: 2,
    height: 5,
    depth: 2,
    color: [0.85, 0.4, 0.4, 1],
  });
  mat4.rotateX(LocalTransform.matrix.getBatch(tiltX), LocalTransform.matrix.getBatch(tiltX), 0.7);
  const tiltY = createRectangle(world, {
    x: 0,
    y: -11,
    z: 3,
    width: 2,
    height: 5,
    depth: 2,
    color: [0.4, 0.85, 0.4, 1],
  });
  mat4.rotateY(LocalTransform.matrix.getBatch(tiltY), LocalTransform.matrix.getBatch(tiltY), 0.7);
  const tiltZ = createRectangle(world, {
    x: -3,
    y: -11,
    z: 3,
    width: 2,
    height: 5,
    depth: 2,
    color: [0.4, 0.4, 0.85, 1],
  });
  mat4.rotateZ(LocalTransform.matrix.getBatch(tiltZ), LocalTransform.matrix.getBatch(tiltZ), 0.7);

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

  return {};
}
