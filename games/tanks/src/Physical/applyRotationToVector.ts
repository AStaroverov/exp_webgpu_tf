import { Vector2 } from '@dimforge/rapier2d-simd/rapier';
import { vec2 } from 'gl-matrix';

const result = vec2.create();
const pivot = vec2.create();

export function applyRotationToVector<T extends Vector2 | vec2>(out: T, vector: Vector2 | vec2, rotation: number): T {
    if (vector instanceof Vector2) {
        result[0] = vector.x;
        result[1] = vector.y;
    } else {
        result[0] = vector[0];
        result[1] = vector[1];
    }

    vec2.rotate(result, result, pivot, rotation);

    if (out instanceof Vector2) {
        out.x = result[0];
        out.y = result[1];
    } else {
        out[0] = result[0];
        out[1] = result[1];
    }
    return out;
}