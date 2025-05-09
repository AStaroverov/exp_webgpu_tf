import * as tf from '@tensorflow/tfjs';
import { InputArrays } from './InputArrays.ts';
import { flatTypedArray } from './flat.ts';
import {
    ALLY_FEATURES_DIM,
    ALLY_SLOTS,
    BATTLE_FEATURES_DIM,
    BULLET_FEATURES_DIM,
    BULLET_SLOTS,
    CONTROLLER_FEATURES_DIM,
    ENEMY_FEATURES_DIM,
    ENEMY_SLOTS,
    TANK_FEATURES_DIM,
} from '../Models/Create.ts';

export function createInputTensors(
    state: InputArrays[],
): tf.Tensor[] {
    //[controller, battleInput, tankInput, enemiesInput, enemiesMaskInput, alliesInput, alliesMaskInput, bulletsInput, bulletsMaskInput],
    return [
        // controller
        tf.tensor2d(flatTypedArray(state.map((s) => s.controllerFeatures)), [state.length, CONTROLLER_FEATURES_DIM]),
        // battle
        tf.tensor2d(flatTypedArray(state.map((s) => s.battleFeatures)), [state.length, BATTLE_FEATURES_DIM]),
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
    ];
}

export function sliceInputTensors(tensors: tf.Tensor[], start: number, size: number): tf.Tensor[] {
    if (tensors.length !== 8) {
        throw new Error('Invalid input tensors length');
    }

    return [
        // controller
        tensors[0].slice([start, 0], [size, -1]),
        // battlefield
        tensors[0].slice([start, 0], [size, -1]),
        // tank
        tensors[1].slice([start, 0], [size, -1]),
        // enemies + mask
        tensors[2].slice([start, 0, 0], [size, -1, -1]),
        tensors[3].slice([start, 0], [size, -1]),
        // allies + mask
        tensors[4].slice([start, 0, 0], [size, -1, -1]),
        tensors[5].slice([start, 0], [size, -1]),
        // bullets + mask
        tensors[6].slice([start, 0, 0], [size, -1, -1]),
        tensors[7].slice([start, 0], [size, -1]),
    ];
}