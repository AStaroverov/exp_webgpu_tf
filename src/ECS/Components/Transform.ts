import { addComponent, defineComponent, Types } from 'bitecs';
import { mat4 } from 'gl-matrix';
import { World } from '../world.ts';

export const LocalTransform = defineComponent({
    matrix: [Types.f32, 16],
});

export const GlobalTransform = defineComponent({
    matrix: [Types.f32, 16],
});

export function addTransformComponents(world: World, id: number) {
    addComponent(world, LocalTransform, id);
    addComponent(world, GlobalTransform, id);
    resetMatrix(LocalTransform.matrix[id]);
    resetMatrix(GlobalTransform.matrix[id]);
}

const tmpTranslate: [number, number, number] = [0, 0, 0];

export function resetMatrix(m: mat4) {
    mat4.identity(m);
}

export function applyMatrixTranslate(m: mat4, x: number, y: number) {
    tmpTranslate[0] = x;
    tmpTranslate[1] = y;
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

export function getMatrixRotationZ(m: mat4) {
    return Math.atan2(m[1], m[0]);
}