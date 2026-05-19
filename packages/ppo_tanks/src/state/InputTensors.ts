import * as tf from '@tensorflow/tfjs';
import {
    ALLY_FEATURES_DIM,
    ALLY_SLOTS,
    BULLET_FEATURES_DIM,
    BULLET_SLOTS,
    ENEMY_FEATURES_DIM,
    ENEMY_SLOTS,
    GRID_CELL_FEATURES,
    GRID_CELLS,
    RAY_FEATURES_DIM,
    RAY_SLOTS,
    TANK_FEATURES_DIM,
    TANK_HISTORY_STEPS,
    TANK_HISTORY_FEATURE_DIM,
    TURRET_FEATURES_DIM,
} from '../models/dims.ts';
import { type InputArrays } from './InputArrays.ts';
import { MAX_TURRETS } from '../../../tanks/src/Plugins/Pilots/Components/TankState.ts';

export { RAY_SLOTS };

function packF32(
    batch: InputArrays[],
    getter: (s: InputArrays) => Float32Array,
    elemSize: number,
): Float32Array {
    const B = batch.length;
    const buf = new Float32Array(B * elemSize);
    for (let b = 0; b < B; b++) {
        buf.set(getter(batch[b]), b * elemSize);
    }
    return buf;
}

function packI32(
    batch: InputArrays[],
    getter: (s: InputArrays) => Int32Array,
    elemSize: number,
): Int32Array {
    const B = batch.length;
    const buf = new Int32Array(B * elemSize);
    for (let b = 0; b < B; b++) {
        buf.set(getter(batch[b]), b * elemSize);
    }
    return buf;
}


export function createInputTensors(
    batch: InputArrays[],
): tf.Tensor[] {
    const B = batch.length;

    return [
        tf.tensor3d(packF32(batch, s => s.tankFeatures, TANK_FEATURES_DIM), [B, 1, TANK_FEATURES_DIM]),
        tf.tensor3d(packF32(batch, s => s.tankFeaturesHistory, TANK_HISTORY_STEPS * TANK_HISTORY_FEATURE_DIM), [B, TANK_HISTORY_STEPS, TANK_HISTORY_FEATURE_DIM]),
        tf.tensor2d(packI32(batch, s => s.tankType, 1), [B, 1]),
        tf.tensor3d(packF32(batch, s => s.turretFeatures, MAX_TURRETS * TURRET_FEATURES_DIM), [B, MAX_TURRETS, TURRET_FEATURES_DIM]),
        tf.tensor3d(packF32(batch, s => s.raysFeatures, RAY_SLOTS * RAY_FEATURES_DIM), [B, RAY_SLOTS, RAY_FEATURES_DIM]),
        tf.tensor3d(packF32(batch, s => s.enemiesFeatures, ENEMY_SLOTS * ENEMY_FEATURES_DIM), [B, ENEMY_SLOTS, ENEMY_FEATURES_DIM]),
        tf.tensor2d(packI32(batch, s => s.enemiesTypes, ENEMY_SLOTS), [B, ENEMY_SLOTS]),
        tf.tensor2d(packF32(batch, s => s.enemiesMask, ENEMY_SLOTS), [B, ENEMY_SLOTS]),
        tf.tensor3d(packF32(batch, s => s.alliesFeatures, ALLY_SLOTS * ALLY_FEATURES_DIM), [B, ALLY_SLOTS, ALLY_FEATURES_DIM]),
        tf.tensor2d(packI32(batch, s => s.alliesTypes, ALLY_SLOTS), [B, ALLY_SLOTS]),
        tf.tensor2d(packF32(batch, s => s.alliesMask, ALLY_SLOTS), [B, ALLY_SLOTS]),
        tf.tensor3d(packF32(batch, s => s.bulletsFeatures, BULLET_SLOTS * BULLET_FEATURES_DIM), [B, BULLET_SLOTS, BULLET_FEATURES_DIM]),
        tf.tensor2d(packF32(batch, s => s.bulletsMask, BULLET_SLOTS), [B, BULLET_SLOTS]),
        tf.tensor3d(packF32(batch, s => s.obstacleGrid, GRID_CELLS * GRID_CELL_FEATURES), [B, GRID_CELLS, GRID_CELL_FEATURES]),
    ];
}
