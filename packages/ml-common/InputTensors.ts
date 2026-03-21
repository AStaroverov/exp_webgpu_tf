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
import { HISTORY_LENGTH } from './historyConfig.ts';
import { type InputArrays, type StateHistory } from './InputArrays.ts';
import { flatTypedArray } from './flat.ts';
import { MAX_TURRETS } from '../tanks/src/Plugins/Pilots/Components/TankState.ts';

export { RAY_SLOTS };

const T = HISTORY_LENGTH;

/**
 * Creates input tensors from state histories.
 * Set-type inputs (rays, enemies, allies, bullets, turrets) are concatenated
 * along the slot dimension across all temporal frames.
 * Tank features become [B, T, TANK_FEATURES_DIM] (T tokens instead of 1).
 */
export function createInputTensors(
    histories: StateHistory[],
): tf.Tensor[] {
    const B = histories.length;

    // Helper: for each batch item, concatenate a field across all T frames
    const concatField = (getter: (s: InputArrays) => Float32Array | Int32Array) => {
        return histories.map(history => flatTypedArray(history.map(getter)));
    };

    return [
        // tank features: [B, T, TANK_FEATURES_DIM]
        tf.tensor3d(
            flatTypedArray(concatField(s => s.tankFeatures) as Float32Array[]),
            [B, T, TANK_FEATURES_DIM],
        ),
        // tank type: [B, T]
        tf.tensor2d(
            flatTypedArray(concatField(s => s.tankType) as Int32Array[]),
            [B, T],
        ),
        // turret features: [B, T * MAX_TURRETS, TURRET_FEATURES_DIM]
        tf.tensor3d(
            flatTypedArray(concatField(s => s.turretFeatures) as Float32Array[]),
            [B, T * MAX_TURRETS, TURRET_FEATURES_DIM],
        ),
        // unified rays: [B, T * RAY_SLOTS, RAY_FEATURES_DIM]
        tf.tensor3d(
            flatTypedArray(concatField(s => s.raysFeatures) as Float32Array[]),
            [B, T * RAY_SLOTS, RAY_FEATURES_DIM],
        ),
        // enemies features: [B, T * ENEMY_SLOTS, ENEMY_FEATURES_DIM]
        tf.tensor3d(
            flatTypedArray(concatField(s => s.enemiesFeatures) as Float32Array[]),
            [B, T * ENEMY_SLOTS, ENEMY_FEATURES_DIM],
        ),
        // enemies types: [B, T * ENEMY_SLOTS]
        tf.tensor2d(
            flatTypedArray(concatField(s => s.enemiesTypes) as Int32Array[]),
            [B, T * ENEMY_SLOTS],
        ),
        // enemies mask: [B, T * ENEMY_SLOTS]
        tf.tensor2d(
            flatTypedArray(concatField(s => s.enemiesMask) as Float32Array[]),
            [B, T * ENEMY_SLOTS],
        ),
        // allies features: [B, T * ALLY_SLOTS, ALLY_FEATURES_DIM]
        tf.tensor3d(
            flatTypedArray(concatField(s => s.alliesFeatures) as Float32Array[]),
            [B, T * ALLY_SLOTS, ALLY_FEATURES_DIM],
        ),
        // allies types: [B, T * ALLY_SLOTS]
        tf.tensor2d(
            flatTypedArray(concatField(s => s.alliesTypes) as Int32Array[]),
            [B, T * ALLY_SLOTS],
        ),
        // allies mask: [B, T * ALLY_SLOTS]
        tf.tensor2d(
            flatTypedArray(concatField(s => s.alliesMask) as Float32Array[]),
            [B, T * ALLY_SLOTS],
        ),
        // bullets features: [B, T * BULLET_SLOTS, BULLET_FEATURES_DIM]
        tf.tensor3d(
            flatTypedArray(concatField(s => s.bulletsFeatures) as Float32Array[]),
            [B, T * BULLET_SLOTS, BULLET_FEATURES_DIM],
        ),
        // bullets mask: [B, T * BULLET_SLOTS]
        tf.tensor2d(
            flatTypedArray(concatField(s => s.bulletsMask) as Float32Array[]),
            [B, T * BULLET_SLOTS],
        ),
    ];
}
