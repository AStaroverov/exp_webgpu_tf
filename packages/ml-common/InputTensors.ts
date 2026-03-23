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
    GRID_SIZE,
    RAY_FEATURES_DIM,
    RAY_SLOTS,
    TANK_FEATURES_DIM,
    TURRET_FEATURES_DIM,
} from '../ml/src/Models/Create.ts';
import { HISTORY_LENGTH } from './historyConfig.ts';
import { type InputArrays, type StateHistory } from './InputArrays.ts';
import { MAX_TURRETS } from '../tanks/src/Plugins/Pilots/Components/TankState.ts';

export { RAY_SLOTS };


// Pack field data from all histories directly into a single flat buffer.
// Eliminates B intermediate arrays per field (was: B+1 allocations → now: 1).
function packF32(
    histories: StateHistory[],
    getter: (s: InputArrays) => Float32Array,
    elemSize: number,
): Float32Array {
    const B = histories.length;
    const T = HISTORY_LENGTH;
    
    const buf = new Float32Array(B * T * elemSize);
    let offset = 0;
    for (let b = 0; b < B; b++) {
        const h = histories[b];
        for (let t = 0; t < T; t++) {
            buf.set(getter(h[t]), offset);
            offset += elemSize;
        }
    }
    return buf;
}

function packI32(
    histories: StateHistory[],
    getter: (s: InputArrays) => Int32Array,
    elemSize: number,
): Int32Array {
    const B = histories.length;
    const T = HISTORY_LENGTH;

    const buf = new Int32Array(B * T * elemSize);
    let offset = 0;
    for (let b = 0; b < B; b++) {
        const h = histories[b];
        for (let t = 0; t < T; t++) {
            buf.set(getter(h[t]), offset);
            offset += elemSize;
        }
    }
    return buf;
}

// Pack obstacle grid with positional features: [obstacle, cell_norm_x, cell_norm_y]
// Static per episode — not multiplied by T.
function packGridF32(histories: StateHistory[]): Float32Array {
    const B = histories.length;
    const buf = new Float32Array(B * GRID_CELLS * GRID_CELL_FEATURES);
    let offset = 0;
    for (let b = 0; b < B; b++) {
        const grid = histories[b][0].obstacleGrid;
        for (let row = 0; row < GRID_SIZE; row++) {
            for (let col = 0; col < GRID_SIZE; col++) {
                buf[offset++] = grid[row * GRID_SIZE + col];
                buf[offset++] = (col + 0.5) / GRID_SIZE - 0.5;
                buf[offset++] = (row + 0.5) / GRID_SIZE - 0.5;
            }
        }
    }
    return buf;
}

export function createInputTensors(
    histories: StateHistory[],
): tf.Tensor[] {
    const T = HISTORY_LENGTH;
    const B = histories.length;

    return [
        tf.tensor3d(packF32(histories, s => s.tankFeatures, TANK_FEATURES_DIM), [B, T, TANK_FEATURES_DIM]),
        tf.tensor2d(packI32(histories, s => s.tankType, 1), [B, T]),
        tf.tensor3d(packF32(histories, s => s.turretFeatures, MAX_TURRETS * TURRET_FEATURES_DIM), [B, T * MAX_TURRETS, TURRET_FEATURES_DIM]),
        tf.tensor3d(packF32(histories, s => s.raysFeatures, RAY_SLOTS * RAY_FEATURES_DIM), [B, T * RAY_SLOTS, RAY_FEATURES_DIM]),
        tf.tensor3d(packF32(histories, s => s.enemiesFeatures, ENEMY_SLOTS * ENEMY_FEATURES_DIM), [B, T * ENEMY_SLOTS, ENEMY_FEATURES_DIM]),
        tf.tensor2d(packI32(histories, s => s.enemiesTypes, ENEMY_SLOTS), [B, T * ENEMY_SLOTS]),
        tf.tensor2d(packF32(histories, s => s.enemiesMask, ENEMY_SLOTS), [B, T * ENEMY_SLOTS]),
        tf.tensor3d(packF32(histories, s => s.alliesFeatures, ALLY_SLOTS * ALLY_FEATURES_DIM), [B, T * ALLY_SLOTS, ALLY_FEATURES_DIM]),
        tf.tensor2d(packI32(histories, s => s.alliesTypes, ALLY_SLOTS), [B, T * ALLY_SLOTS]),
        tf.tensor2d(packF32(histories, s => s.alliesMask, ALLY_SLOTS), [B, T * ALLY_SLOTS]),
        tf.tensor3d(packF32(histories, s => s.bulletsFeatures, BULLET_SLOTS * BULLET_FEATURES_DIM), [B, T * BULLET_SLOTS, BULLET_FEATURES_DIM]),
        tf.tensor2d(packF32(histories, s => s.bulletsMask, BULLET_SLOTS), [B, T * BULLET_SLOTS]),
        tf.tensor3d(packGridF32(histories), [B, GRID_CELLS, GRID_CELL_FEATURES]),
    ];
}
