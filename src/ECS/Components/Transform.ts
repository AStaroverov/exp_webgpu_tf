import { defineComponent, Types } from 'bitecs';
import { mat4 } from 'gl-matrix';

export const Transform = defineComponent({
    matrix: [Types.f32, 16],
});

export function resetMatrix(id: number) {
    mat4.identity(Transform.matrix[id]);
}

const tmpTranslate: [number, number, number] = [0, 0, 0];

export function applyMatrixTranslate(id: number, x: number, y: number) {
    tmpTranslate[0] = x;
    tmpTranslate[1] = y;
    mat4.translate(Transform.matrix[id], Transform.matrix[id], tmpTranslate);
}

export function setMatrixTranslate(id: number, x: number, y: number, z?: number) {
    Transform.matrix[id][12] = x; // Set translation X
    Transform.matrix[id][13] = y; // Set translation Y
    Transform.matrix[id][14] = z ?? Transform.matrix[id][14]; // Set translation Z
}

export function setMatrix(id: number, matrix: number[] | Float32Array) {
    Transform.matrix[id].set(matrix);
}
