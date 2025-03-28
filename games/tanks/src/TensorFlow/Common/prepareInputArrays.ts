import { clamp } from 'lodash-es';
import { shuffle } from '../../../../../lib/shuffle.ts';
import {
    TANK_INPUT_TENSOR_BULLET_BUFFER,
    TANK_INPUT_TENSOR_ENEMY_BUFFER,
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';
import { TANK_FEATURES_DIM } from './models.ts';

function normForTanh(v: number, size: number): number {
    return v / size;
}

function normForRelu(v: number, size: number): number {
    return (normForTanh(v, size) + 1) / 2;
}

function norm(v: number, size: number): number {
    return clamp(normForRelu(v, size) * 0.6 + 0.2, 0, 1);
}

// Для случайного распределения врагов/пуль
const ENEMIES_INDEXES = new Uint32Array(Array.from({ length: TANK_INPUT_TENSOR_MAX_ENEMIES }, (_, i) => i));
const BULLETS_INDEXES = new Uint32Array(Array.from({ length: TANK_INPUT_TENSOR_MAX_BULLETS }, (_, i) => i));

/**
 * Возвращает три массива:
 *  - tankFeatures: Float32Array длины 7 (примерно)
 *  - enemiesFeatures: Float32Array длины [TANK_INPUT_TENSOR_MAX_ENEMIES * TANK_INPUT_TENSOR_ENEMY_BUFFER]
 *  - bulletsFeatures: Float32Array длины [TANK_INPUT_TENSOR_MAX_BULLETS * TANK_INPUT_TENSOR_BULLET_BUFFER]
 *
 * Но семантически мы их будем интерпретировать как:
 *   enemiesFeatures -> 2D: [TANK_INPUT_TENSOR_MAX_ENEMIES, TANK_INPUT_TENSOR_ENEMY_BUFFER]
 *   bulletsFeatures -> 2D: [TANK_INPUT_TENSOR_MAX_BULLETS, TANK_INPUT_TENSOR_BULLET_BUFFER]
 */

export type InputArrays = {
    tankFeatures: Float32Array,
    enemiesFeatures: Float32Array,
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
    //    Интерпретируем как матрицу [TANK_INPUT_TENSOR_MAX_ENEMIES, TANK_INPUT_TENSOR_ENEMY_BUFFER]
    const enemiesFeatures = new Float32Array(
        TANK_INPUT_TENSOR_MAX_ENEMIES * TANK_INPUT_TENSOR_ENEMY_BUFFER,
    );
    const enemiesBuffer = TankInputTensor.enemiesData.getBatch(tankEid);

    // Перемешиваем индексы, чтобы случайным образом «раскидать» врагов
    shuffle(ENEMIES_INDEXES);

    for (let r = 0; r < TANK_INPUT_TENSOR_MAX_ENEMIES; r++) {
        const w = ENEMIES_INDEXES[r]; // случайный слот, куда положим r-го врага из source
        const srcOffset = r * TANK_INPUT_TENSOR_ENEMY_BUFFER;
        const dstOffset = w * TANK_INPUT_TENSOR_ENEMY_BUFFER;

        // если во входном буфере признак presence = 0, значит врага нет
        if (enemiesBuffer[srcOffset] === 0) {
            continue;
        }

        // presence
        enemiesFeatures[dstOffset + 0] = 1;
        // остальные поля (X, Y, width, height, speedX, speedY...) — как в вашем коде
        enemiesFeatures[dstOffset + 1] = norm(enemiesBuffer[srcOffset + 1] - tankX, width);
        enemiesFeatures[dstOffset + 2] = norm(enemiesBuffer[srcOffset + 2] - tankY, height);
        enemiesFeatures[dstOffset + 3] = norm(enemiesBuffer[srcOffset + 3], width);
        enemiesFeatures[dstOffset + 4] = norm(enemiesBuffer[srcOffset + 4], height);
        enemiesFeatures[dstOffset + 5] = norm(enemiesBuffer[srcOffset + 5] - tankX, width);
        enemiesFeatures[dstOffset + 6] = norm(enemiesBuffer[srcOffset + 6] - tankY, height);
    }

    // 3) Аналогично — массив для пуль
    //    [TANK_INPUT_TENSOR_MAX_BULLETS, TANK_INPUT_TENSOR_BULLET_BUFFER]
    const bulletsFeatures = new Float32Array(
        TANK_INPUT_TENSOR_MAX_BULLETS * TANK_INPUT_TENSOR_BULLET_BUFFER,
    );
    const bulletsBuffer = TankInputTensor.bulletsData.getBatch(tankEid);
    shuffle(BULLETS_INDEXES);

    for (let r = 0; r < TANK_INPUT_TENSOR_MAX_BULLETS; r++) {
        const w = BULLETS_INDEXES[r];
        const srcOffset = r * TANK_INPUT_TENSOR_BULLET_BUFFER;
        const dstOffset = w * TANK_INPUT_TENSOR_BULLET_BUFFER;

        if (bulletsBuffer[srcOffset] === 0) {
            continue;
        }

        // presence
        bulletsFeatures[dstOffset + 0] = 1;
        // остальные поля — аналогично
        bulletsFeatures[dstOffset + 1] = norm(bulletsBuffer[srcOffset + 1] - tankX, width);
        bulletsFeatures[dstOffset + 2] = norm(bulletsBuffer[srcOffset + 2] - tankY, height);
        bulletsFeatures[dstOffset + 3] = norm(bulletsBuffer[srcOffset + 3], width);
        bulletsFeatures[dstOffset + 4] = norm(bulletsBuffer[srcOffset + 4], height);
    }

    return {
        tankFeatures,
        enemiesFeatures,
        bulletsFeatures,
    };
}
