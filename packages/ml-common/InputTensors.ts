import * as tf from '@tensorflow/tfjs';
import {
    ALLY_FEATURES_DIM,
    ALLY_SLOTS,
    BULLET_FEATURES_DIM,
    BULLET_SLOTS,
    ENEMY_FEATURES_DIM,
    ENEMY_SLOTS,
    RAY_FEATURES_DIM,
    RAY_SLOTS,
    TANK_FEATURES_DIM,
    TURRET_FEATURES_DIM,
} from '../ml/src/Models/Create.ts';
import { InputArrays } from './InputArrays.ts';
import { flatTypedArray } from './flat.ts';
import { MAX_TURRETS } from '../tanks/src/Pilots/Components/TankState.ts';

export { RAY_SLOTS };

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
        // turret features
        tf.tensor3d(flatTypedArray(state.map((s) => s.turretFeatures)), [state.length, MAX_TURRETS, TURRET_FEATURES_DIM]),
        // unified rays (environment + turret)
        tf.tensor3d(
            flatTypedArray(state.map((s) => s.raysFeatures)),
            [state.length, RAY_SLOTS, RAY_FEATURES_DIM],
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
        )
    ];
}
