import { addComponent } from 'bitecs';
import { mat4 } from 'gl-matrix';
import { World } from '../world.ts';
import { NestedArray } from '../../utils.ts';
import { delegate } from '../../delegate.ts';

export const LocalTransform = ({
    matrix: NestedArray.f32(16, delegate.defaultSize),
});

export const GlobalTransform = ({
    matrix: NestedArray.f32(16, delegate.defaultSize),
});

export function addTransformComponents(world: World, id: number) {
    addComponent(world, id, LocalTransform);
    addComponent(world, id, GlobalTransform);
    LocalTransform.matrix.setBatch(id, IDENTIFY_MATRIX);
    GlobalTransform.matrix.setBatch(id, IDENTIFY_MATRIX);
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

export function setMatrixTranslate(m: mat4, x: number, y: number, z?: number) {
    m[12] = x; // Set translation X
    m[13] = y; // Set translation Y
    m[14] = z ?? m[14]; // Set translation Z
}

export function setMatrixRotateZ(m: mat4, angle: number) {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    m[0] = cos; // Set rotation X
    m[1] = sin; // Set rotation Y
    m[4] = -sin; // Set rotation X
    m[5] = cos; // Set rotation Y
}

export function getMatrixTranslationX(m: mat4) {
    return m[12];
}

export function getMatrixTranslationY(m: mat4) {
    return m[13];
}

export function getMatrixTranslation(m: mat4) {
    return (m as Float32Array).subarray(12, 14);
}

export function getMatrixRotationZ(m: mat4) {
    return Math.atan2(m[1], m[0]);
}

export function applyMatrixScale(m: mat4, sx: number, sy: number, sz: number = 1) {
    mat4.scale(m, m, [sx, sy, sz]);
}

export function setMatrixScale(m: mat4, sx: number, sy: number, sz: number = 1) {
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