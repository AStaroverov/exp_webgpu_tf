import { addComponent, World } from "bitecs";
import { mat4 } from "gl-matrix";
import { NestedArray } from "../../utils.ts";
import { delegate } from "../../../../common/src/delegate.ts";

export function createLocalTransformComponent() {
  return {
    matrix: NestedArray.f32(16, delegate.defaultSize),
  };
}

export function createGlobalTransformComponent() {
  return {
    matrix: NestedArray.f32(16, delegate.defaultSize),
  };
}

export function addTransformComponents(world: World, id: number) {
  // Each world owns its OWN transform instances (created per-world in
  // createRenderComponents). There is no module-singleton fallback: a world
  // that lacks them is a bug, not a default case — and across a real worker the
  // module re-executes over private memory, so a shared singleton would silently
  // break. Read strictly from world.components.
  const components = (
    world as World<{
      components?: {
        LocalTransform: ReturnType<typeof createLocalTransformComponent>;
        GlobalTransform: ReturnType<typeof createGlobalTransformComponent>;
      };
    }>
  ).components;
  if (!components?.LocalTransform || !components?.GlobalTransform) {
    throw new Error(
      "addTransformComponents: world.components.LocalTransform/GlobalTransform are missing",
    );
  }
  const worldLocalTransform = components.LocalTransform;
  const worldGlobalTransform = components.GlobalTransform;
  addComponent(world, id, worldLocalTransform);
  addComponent(world, id, worldGlobalTransform);
  worldLocalTransform.matrix.setBatch(id, IDENTIFY_MATRIX);
  worldGlobalTransform.matrix.setBatch(id, IDENTIFY_MATRIX);
}

const IDENTIFY_MATRIX: mat4 = mat4.create();

const tmpTranslate: [number, number, number] = [0, 0, 0];

export function applyMatrixTranslate(m: mat4, x: number, y: number, z: number) {
  tmpTranslate[0] = x;
  tmpTranslate[1] = y;
  tmpTranslate[2] = z;
  mat4.translate(m, m, tmpTranslate);
}

export function applyMatrixRotateZ(m: mat4, angle: number) {
  mat4.rotateZ(m, m, angle);
}

export function setMatrixTranslate(m: mat4, x: number, y: number, z: number) {
  m[12] = x;
  m[13] = y;
  m[14] = z;
}

export function setMatrixRotateZ(m: mat4, angle: number) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const sx = Math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
  const sy = Math.sqrt(m[4] * m[4] + m[5] * m[5] + m[6] * m[6]);
  m[0] = cos * sx;
  m[1] = sin * sx;
  m[2] = 0;
  m[4] = -sin * sy;
  m[5] = cos * sy;
  m[6] = 0;
}

export function setMatrixRotateY(m: mat4, angle: number) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const sx = Math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
  const sz = Math.sqrt(m[8] * m[8] + m[9] * m[9] + m[10] * m[10]);
  m[0] = cos * sx;
  m[1] = 0;
  m[2] = -sin * sx;
  m[8] = sin * sz;
  m[9] = 0;
  m[10] = cos * sz;
}

export function setMatrixRotateX(m: mat4, angle: number) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const sy = Math.sqrt(m[4] * m[4] + m[5] * m[5] + m[6] * m[6]);
  const sz = Math.sqrt(m[8] * m[8] + m[9] * m[9] + m[10] * m[10]);
  m[4] = 0;
  m[5] = cos * sy;
  m[6] = sin * sy;
  m[8] = 0;
  m[9] = -sin * sz;
  m[10] = cos * sz;
}

export function getMatrixTranslationX(m: mat4) {
  return m[12];
}

export function getMatrixTranslationY(m: mat4) {
  return m[13];
}

export function getMatrixTranslation(m: mat4) {
  return (m as Float32Array).subarray(12, 15);
}

export function getMatrixRotationZ(m: mat4) {
  return Math.atan2(m[1], m[0]);
}

export function applyMatrixScale(m: mat4, sx: number, sy = sx, sz = sx) {
  mat4.scale(m, m, [sx, sy, sz]);
}

export function setMatrixScale(m: mat4, sx: number, sy = sx, sz = sx) {
  // Extract current rotation
  const cosR = m[0];
  const sinR = m[1];
  const len = Math.sqrt(cosR * cosR + sinR * sinR);
  const cos = len > 0 ? cosR / len : 1;
  const sin = len > 0 ? sinR / len : 0;

  // Apply scale with rotation preserved
  m[0] = cos * sx;
  m[1] = sin * sx;
  m[4] = -sin * sy;
  m[5] = cos * sy;
  m[10] = sz;
}

export function getMatrixScaleX(m: mat4) {
  return Math.sqrt(m[0] * m[0] + m[1] * m[1]);
}

export function getMatrixScaleY(m: mat4) {
  return Math.sqrt(m[4] * m[4] + m[5] * m[5]);
}
