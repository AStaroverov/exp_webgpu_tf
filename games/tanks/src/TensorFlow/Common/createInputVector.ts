import { INPUT_DIM } from './consts.ts';
import {
    TANK_INPUT_TENSOR_BULLET_BUFFER,
    TANK_INPUT_TENSOR_ENEMY_BUFFER,
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';
import { clamp } from 'lodash-es';
import { shuffle } from '../../../../../lib/shuffle.ts';

const enemiesNormalizedBuffer = new Float32Array(TANK_INPUT_TENSOR_ENEMY_BUFFER * TANK_INPUT_TENSOR_MAX_ENEMIES);
const bulletsNormalizedBuffer = new Float32Array(TANK_INPUT_TENSOR_BULLET_BUFFER * TANK_INPUT_TENSOR_MAX_BULLETS);

function normForTanh(v: number, size: number): number {
    return v / size;
}

function normForRelu(v: number, size: number): number {
    return (normForTanh(v, size) + 1) / 2;
}

function norm(v: number, size: number): number {
    return clamp(normForRelu(v, size) * 0.6 + 0.2, 0, 1);
}

const ENEMIES_INDEXES = new Uint32Array(Array.from({ length: TANK_INPUT_TENSOR_MAX_ENEMIES }, (_, i) => i));
const BULLETS_INDEXES = new Uint32Array(Array.from({ length: TANK_INPUT_TENSOR_MAX_BULLETS }, (_, i) => i));

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
    shuffle(ENEMIES_INDEXES);
    const enemiesBuffer = TankInputTensor.enemiesData.getBatch(tankEid);
    for (let r = 0, l = TANK_INPUT_TENSOR_ENEMY_BUFFER; r < TANK_INPUT_TENSOR_MAX_ENEMIES; r++) {
        const w = ENEMIES_INDEXES[r];
        if (enemiesBuffer[r * l] === 0) continue;
        enemiesNormalizedBuffer[w * l + 0] = 1;
        enemiesNormalizedBuffer[w * l + 1] = norm(enemiesBuffer[r * l + 1] - tankX, width);
        enemiesNormalizedBuffer[w * l + 2] = norm(enemiesBuffer[r * l + 2] - tankY, height);
        enemiesNormalizedBuffer[w * l + 3] = norm(enemiesBuffer[r * l + 3], width);
        enemiesNormalizedBuffer[w * l + 4] = norm(enemiesBuffer[r * l + 4], height);
        enemiesNormalizedBuffer[w * l + 5] = norm(enemiesBuffer[r * l + 5] - tankX, width);
        enemiesNormalizedBuffer[w * l + 6] = norm(enemiesBuffer[r * l + 6] - tankY, height);
    }
    inputVector.set(enemiesNormalizedBuffer, k);
    k += enemiesNormalizedBuffer.length;

    // Bullets data
    shuffle(BULLETS_INDEXES);
    const bulletsBuffer = TankInputTensor.bulletsData.getBatch(tankEid);
    for (let r = 0, l = TANK_INPUT_TENSOR_BULLET_BUFFER; r < TANK_INPUT_TENSOR_MAX_BULLETS; r++) {
        const w = BULLETS_INDEXES[r];
        if (bulletsBuffer[r * l] === 0) continue;
        bulletsNormalizedBuffer[w * l + 0] = 1;
        bulletsNormalizedBuffer[w * l + 1] = norm(bulletsBuffer[r * l + 1] - tankX, width);
        bulletsNormalizedBuffer[w * l + 2] = norm(bulletsBuffer[r * l + 2] - tankY, height);
        bulletsNormalizedBuffer[w * l + 3] = norm(bulletsBuffer[r * l + 3], width);
        bulletsNormalizedBuffer[w * l + 4] = norm(bulletsBuffer[r * l + 4], height);
    }
    inputVector.set(bulletsNormalizedBuffer, k);
    k += bulletsNormalizedBuffer.length;

    return inputVector;
}