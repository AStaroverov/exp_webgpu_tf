import { clamp } from 'lodash-es';
import { shuffle } from '../../../../../lib/shuffle.ts';
import { BULLET_BUFFER, ENEMY_BUFFER, TankInputTensor } from '../../ECS/Components/TankState.ts';
import { BULLET_FEATURES_DIM, BULLET_SLOTS, ENEMY_FEATURES_DIM, ENEMY_SLOTS, TANK_FEATURES_DIM } from './models.ts';

function normForTanh(v: number, size: number): number {
    return v / size;
}

function normForRelu(v: number, size: number): number {
    return (normForTanh(v, size) + 1) / 2;
}

function norm(v: number, size: number): number {
    return clamp(normForRelu(v, size) * 1 + 1, 0, 3);
}

// Для случайного распределения врагов/пуль
const ENEMIES_INDEXES = new Uint32Array(Array.from({ length: ENEMY_SLOTS }, (_, i) => i));
const BULLETS_INDEXES = new Uint32Array(Array.from({ length: BULLET_SLOTS }, (_, i) => i));

/**
 * Возвращает три массива:
 *  - tankFeatures: Float32Array длины 7 (примерно)
 *  - enemiesFeatures: Float32Array длины [ENEMY_SLOTS * ENEMY_FEATURES_DIM]
 *  - bulletsFeatures: Float32Array длины [BULLET_SLOTS * BULLET_FEATURES_DIM]
 *
 * Но семантически мы их будем интерпретировать как:
 *   enemiesFeatures -> 2D: [ENEMY_SLOTS, ENEMY_FEATURES_DIM]
 *   bulletsFeatures -> 2D: [BULLET_SLOTS, BULLET_FEATURES_DIM]
 */

export type InputArrays = {
    tankFeatures: Float32Array,
    enemiesMask: Float32Array,
    enemiesFeatures: Float32Array,
    bulletsMask: Float32Array,
    bulletsFeatures: Float32Array
}

export function prepareInputArrays(
    tankEid: number,
    width: number,
    height: number,
): InputArrays {
    // 1) Готовим массив для танка
    const tankFeatures = new Float32Array(TANK_FEATURES_DIM);

    const tankX = TankInputTensor.position.get(tankEid, 0);
    const tankY = TankInputTensor.position.get(tankEid, 1);
    const speedX = TankInputTensor.speed.get(tankEid, 0);
    const speedY = TankInputTensor.speed.get(tankEid, 1);
    const turretTargetX = TankInputTensor.turretTarget.get(tankEid, 0);
    const turretTargetY = TankInputTensor.turretTarget.get(tankEid, 1);

    let k = 0;
    tankFeatures[k++] = TankInputTensor.health[tankEid];                 // [0]  здоровье
    tankFeatures[k++] = norm(tankX - width / 2, width);                  // [1]  норм. позиция X
    tankFeatures[k++] = norm(tankY - height / 2, height);                // [2]  норм. позиция Y
    tankFeatures[k++] = norm(speedX, width);                             // [3]  норм. скорость X
    tankFeatures[k++] = norm(speedY, height);                            // [4]  норм. скорость Y
    tankFeatures[k++] = norm(turretTargetX - tankX, width);              // [5]  норма. наводка башни X
    tankFeatures[k++] = norm(turretTargetY - tankY, height);             // [6]  норма. наводка башни Y

    // 2) Массив для врагов
    //    Интерпретируем как матрицу [ENEMY_SLOTS, ENEMY_FEATURES_DIM]
    const enemiesMask = new Float32Array(ENEMY_SLOTS);
    const enemiesFeatures = new Float32Array(ENEMY_SLOTS * ENEMY_FEATURES_DIM);
    const enemiesBuffer = TankInputTensor.enemiesData.getBatch(tankEid);

    shuffle(ENEMIES_INDEXES);

    for (let r = 0; r < ENEMY_SLOTS; r++) {
        const w = ENEMIES_INDEXES[r];
        const dstOffset = w * ENEMY_BUFFER;
        const srcOffset = r * ENEMY_BUFFER;

        // если во входном буфере признак presence = 0, значит врага нет
        if (enemiesBuffer[srcOffset] === 0) {
            continue;
        }

        enemiesMask[w] = 1;
        // остальные поля (X, Y, width, height, speedX, speedY...)
        enemiesFeatures[dstOffset + 0] = norm(enemiesBuffer[srcOffset + 1] - tankX, width);
        enemiesFeatures[dstOffset + 1] = norm(enemiesBuffer[srcOffset + 2] - tankY, height);
        enemiesFeatures[dstOffset + 2] = norm(enemiesBuffer[srcOffset + 3], width);
        enemiesFeatures[dstOffset + 3] = norm(enemiesBuffer[srcOffset + 4], height);
        enemiesFeatures[dstOffset + 4] = norm(enemiesBuffer[srcOffset + 5] - tankX, width);
        enemiesFeatures[dstOffset + 5] = norm(enemiesBuffer[srcOffset + 6] - tankY, height);
    }

    // 3) Аналогично — массив для пуль
    //    [BULLET_SLOTS, BULLET_FEATURES_DIM]
    const bulletsMask = new Float32Array(BULLET_SLOTS);
    const bulletsFeatures = new Float32Array(BULLET_SLOTS * BULLET_FEATURES_DIM);
    const bulletsBuffer = TankInputTensor.bulletsData.getBatch(tankEid);

    shuffle(BULLETS_INDEXES);

    for (let r = 0; r < BULLET_SLOTS; r++) {
        const w = BULLETS_INDEXES[r];
        const dstOffset = w * BULLET_BUFFER;
        const srcOffset = r * BULLET_BUFFER;

        if (bulletsBuffer[srcOffset] === 0) {
            continue;
        }

        bulletsMask[w] = 1;
        // остальные поля — аналогично
        bulletsFeatures[dstOffset + 0] = norm(bulletsBuffer[srcOffset + 1] - tankX, width);
        bulletsFeatures[dstOffset + 1] = norm(bulletsBuffer[srcOffset + 2] - tankY, height);
        bulletsFeatures[dstOffset + 2] = norm(bulletsBuffer[srcOffset + 3], width);
        bulletsFeatures[dstOffset + 3] = norm(bulletsBuffer[srcOffset + 4], height);
    }

    return {
        tankFeatures,
        enemiesMask,
        enemiesFeatures,
        bulletsMask,
        bulletsFeatures,
    };
}
