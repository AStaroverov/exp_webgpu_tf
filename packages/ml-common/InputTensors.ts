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
        // tank features + type
        tf.tensor2d(flatTypedArray(state.map((s) => s.tankFeatures)), [state.length, TANK_FEATURES_DIM]),
        tf.tensor2d(
            flatTypedArray(state.map((s) => s.tankType)),
            [state.length, 1],
        ),
        // enemies features + types + mask
        tf.tensor3d(
            flatTypedArray(state.map((s) => s.enemiesFeatures)),
            [state.length, ENEMY_SLOTS, ENEMY_FEATURES_DIM],
        ),
        tf.tensor2d(
            flatTypedArray(state.map((s) => s.enemiesTypes)),
            [state.length, ENEMY_SLOTS],
        ),
        tf.tensor2d(
            flatTypedArray(state.map((s) => s.enemiesMask)),
            [state.length, ENEMY_SLOTS],
        ),
        // allies features + types + mask
        tf.tensor3d(
            flatTypedArray(state.map((s) => s.alliesFeatures)),
            [state.length, ALLY_SLOTS, ALLY_FEATURES_DIM],
        ),
        tf.tensor2d(
            flatTypedArray(state.map((s) => s.alliesTypes)),
            [state.length, ALLY_SLOTS],
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

const INPUT_TENSORS_COUNT = 14;

export function sliceInputTensors(tensors: tf.Tensor[], start: number, size: number): tf.Tensor[] {
    if (tensors.length !== INPUT_TENSORS_COUNT) {
        throw new Error(`Invalid input tensors length: expected ${INPUT_TENSORS_COUNT}, got ${tensors.length}`);
    }

    return [
        // tank features + type
        tensors[0].slice([start, 0], [size, -1]),
        tensors[1].slice([start, 0], [size, -1]),
        // enemies features + types + mask
        tensors[2].slice([start, 0, 0], [size, -1, -1]),
        tensors[3].slice([start, 0], [size, -1]),
        tensors[4].slice([start, 0], [size, -1]),
        // allies features + types + mask
        tensors[5].slice([start, 0, 0], [size, -1, -1]),
        tensors[6].slice([start, 0], [size, -1]),
        tensors[7].slice([start, 0], [size, -1]),
        // bullets + mask
        tensors[8].slice([start, 0, 0], [size, -1, -1]),
        tensors[9].slice([start, 0], [size, -1]),
        // environment rays features + hit types
        tensors[10].slice([start, 0, 0], [size, -1, -1]),
        tensors[11].slice([start, 0], [size, -1]),
        // turret rays features + hit types
        tensors[12].slice([start, 0, 0], [size, -1, -1]),
        tensors[13].slice([start, 0], [size, -1]),
    ];
}
