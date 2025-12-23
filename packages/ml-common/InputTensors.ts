import * as tf from '@tensorflow/tfjs';
import {
    ALLY_FEATURES_DIM,
    ALLY_SLOTS,
    BULLET_FEATURES_DIM,
    BULLET_SLOTS,
    ENEMY_FEATURES_DIM,
    ENEMY_SLOTS,
    ENV_RAY_FEATURES_DIM,
    ENV_RAY_SLOTS,
    TANK_FEATURES_DIM,
    TURRET_RAY_FEATURES_DIM,
    TURRET_RAY_SLOTS,
} from '../ml/src/Models/Create.ts';
import { InputArrays } from './InputArrays.ts';
import { flatTypedArray } from './flat.ts';

export function createInputTensors(
    state: InputArrays[],
): tf.Tensor[] {
    return [
        // tank
        tf.tensor2d(flatTypedArray(state.map((s) => s.tankFeatures)), [state.length, TANK_FEATURES_DIM]),
        // enemies + mask
        tf.tensor3d(
            flatTypedArray(state.map((s) => s.enemiesFeatures)),
            [state.length, ENEMY_SLOTS, ENEMY_FEATURES_DIM],
        ),
        tf.tensor2d(
            flatTypedArray(state.map((s) => s.enemiesMask)),
            [state.length, ENEMY_SLOTS],
        ),
        // allies + mask
        tf.tensor3d(
            flatTypedArray(state.map((s) => s.alliesFeatures)),
            [state.length, ALLY_SLOTS, ALLY_FEATURES_DIM],
        ),
        tf.tensor2d(
            flatTypedArray(state.map((s) => s.alliesMask)),
            [state.length, ALLY_SLOTS],
        ),
        // bullets + mask
        tf.tensor3d(
            flatTypedArray(state.map((s) => s.bulletsFeatures)),
            [state.length, BULLET_SLOTS, BULLET_FEATURES_DIM],
        ),
        tf.tensor2d(
            flatTypedArray(state.map((s) => s.bulletsMask)),
            [state.length, BULLET_SLOTS],
        ),
        // environment rays features + hit types
        tf.tensor3d(
            flatTypedArray(state.map((s) => s.envRaysFeatures)),
            [state.length, ENV_RAY_SLOTS, ENV_RAY_FEATURES_DIM],
        ),
        tf.tensor2d(
            flatTypedArray(state.map((s) => s.envRaysTypes)),
            [state.length, ENV_RAY_SLOTS],
        ),
        // turret rays features + hit types
        tf.tensor3d(
            flatTypedArray(state.map((s) => s.turretRaysFeatures)),
            [state.length, TURRET_RAY_SLOTS, TURRET_RAY_FEATURES_DIM],
        ),
        tf.tensor2d(
            flatTypedArray(state.map((s) => s.turretRaysTypes)),
            [state.length, TURRET_RAY_SLOTS],
        ),
    ];
}

const INPUT_TENSORS_COUNT = 11;

export function sliceInputTensors(tensors: tf.Tensor[], start: number, size: number): tf.Tensor[] {
    if (tensors.length !== INPUT_TENSORS_COUNT) {
        throw new Error(`Invalid input tensors length: expected ${INPUT_TENSORS_COUNT}, got ${tensors.length}`);
    }

    return [
        // tank
        tensors[0].slice([start, 0], [size, -1]),
        // enemies + mask
        tensors[1].slice([start, 0, 0], [size, -1, -1]),
        tensors[2].slice([start, 0], [size, -1]),
        // allies + mask
        tensors[3].slice([start, 0, 0], [size, -1, -1]),
        tensors[4].slice([start, 0], [size, -1]),
        // bullets + mask
        tensors[5].slice([start, 0, 0], [size, -1, -1]),
        tensors[6].slice([start, 0], [size, -1]),
        // environment rays features + hit types
        tensors[7].slice([start, 0, 0], [size, -1, -1]),
        tensors[8].slice([start, 0], [size, -1]),
        // turret rays features + hit types
        tensors[9].slice([start, 0, 0], [size, -1, -1]),
        tensors[10].slice([start, 0], [size, -1]),
    ];
}
