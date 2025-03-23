import { INPUT_DIM } from './consts.ts';
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';
import { clamp } from 'lodash-es';

const enemiesNormalizedBuffer = new Float32Array(7 * TANK_INPUT_TENSOR_MAX_ENEMIES);
const bulletsNormalizedBuffer = new Float32Array(5 * TANK_INPUT_TENSOR_MAX_BULLETS);

function normForTanh(v: number, size: number): number {
    return v / size;
}

function normForRelu(v: number, size: number): number {
    return (normForTanh(v, size) + 1) / 2;
}

function norm(v: number, size: number): number {
    return clamp(normForRelu(v, size) * 0.6 + 0.2, 0, 1);
}

export function createInputVector(tankEid: number, width: number, height: number) {
    enemiesNormalizedBuffer.fill(0);
    bulletsNormalizedBuffer.fill(0);

    const inputVector = new Float32Array(INPUT_DIM);
    const tankX = TankInputTensor.position.get(tankEid, 0);
    const tankY = TankInputTensor.position.get(tankEid, 1);
    const speedX = TankInputTensor.speed.get(tankEid, 0);
    const speedY = TankInputTensor.speed.get(tankEid, 1);
    const turretTargetX = TankInputTensor.turretTarget.get(tankEid, 0);
    const turretTargetY = TankInputTensor.turretTarget.get(tankEid, 1);
    let k = 0;

    // Tank state
    inputVector[k++] = TankInputTensor.health[tankEid];
    inputVector[k++] = norm(tankX - width / 2, width);
    inputVector[k++] = norm(tankY - height / 2, height);
    inputVector[k++] = norm(speedX, width);
    inputVector[k++] = norm(speedY, height);
    inputVector[k++] = norm(turretTargetX - tankX, width);
    inputVector[k++] = norm(turretTargetY - tankY, height);

    // Enemies data
    const enemiesBuffer = TankInputTensor.enemiesData.getBatche(tankEid);
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_ENEMIES; i++) {
        if (enemiesBuffer[i * 7] === 0) continue;
        enemiesNormalizedBuffer[i * 7 + 0] = 1;
        enemiesNormalizedBuffer[i * 7 + 1] = norm(enemiesBuffer[i * 7 + 1] - tankX, width);
        enemiesNormalizedBuffer[i * 7 + 2] = norm(enemiesBuffer[i * 7 + 2] - tankY, height);
        enemiesNormalizedBuffer[i * 7 + 3] = norm(enemiesBuffer[i * 7 + 3], width);
        enemiesNormalizedBuffer[i * 7 + 4] = norm(enemiesBuffer[i * 7 + 4], height);
        enemiesNormalizedBuffer[i * 7 + 5] = norm(enemiesBuffer[i * 7 + 5] - tankX, width);
        enemiesNormalizedBuffer[i * 7 + 6] = norm(enemiesBuffer[i * 7 + 6] - tankY, height);
    }
    inputVector.set(enemiesNormalizedBuffer, k);
    k += enemiesNormalizedBuffer.length;

    // Bullets data
    const bulletsBuffer = TankInputTensor.bulletsData.getBatche(tankEid);
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_BULLETS; i++) {
        if (bulletsBuffer[i * 5] === 0) continue;
        bulletsNormalizedBuffer[i * 5 + 0] = 1;
        bulletsNormalizedBuffer[i * 5 + 1] = norm(bulletsBuffer[i * 5 + 1] - tankX, width);
        bulletsNormalizedBuffer[i * 5 + 2] = norm(bulletsBuffer[i * 5 + 2] - tankY, height);
        bulletsNormalizedBuffer[i * 5 + 3] = norm(bulletsBuffer[i * 5 + 3], width);
        bulletsNormalizedBuffer[i * 5 + 4] = norm(bulletsBuffer[i * 5 + 4], height);
    }
    inputVector.set(bulletsNormalizedBuffer, k);
    k += bulletsNormalizedBuffer.length;

    return inputVector;
}