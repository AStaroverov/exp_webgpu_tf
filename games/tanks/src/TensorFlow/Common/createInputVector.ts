import { INPUT_DIM } from './consts.ts';
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';

const enemiesNormalizedBuffer = new Float32Array(6 * TANK_INPUT_TENSOR_MAX_ENEMIES);
const bulletsNormalizedBuffer = new Float32Array(5 * TANK_INPUT_TENSOR_MAX_BULLETS);

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
    inputVector[k++] = (tankX / width) * 2 - 1;
    inputVector[k++] = (tankY / height) * 2 - 1;
    inputVector[k++] = speedX / width;
    inputVector[k++] = speedY / height;
    inputVector[k++] = ((turretTargetX - tankX) / width);
    inputVector[k++] = ((turretTargetY - tankY) / height);

    // Enemies data
    const enemiesBuffer = TankInputTensor.enemiesData.getBatche(tankEid);
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_ENEMIES; i++) {
        if (enemiesBuffer[i * 6] === 0) continue;
        enemiesNormalizedBuffer[i * 6 + 0] = 1;
        enemiesNormalizedBuffer[i * 6 + 1] = ((enemiesBuffer[i * 6 + 1] - tankX) / width);
        enemiesNormalizedBuffer[i * 6 + 2] = ((enemiesBuffer[i * 6 + 2] - tankY) / height);
        enemiesNormalizedBuffer[i * 6 + 3] = enemiesBuffer[i * 6 + 3] / width;
        enemiesNormalizedBuffer[i * 6 + 4] = enemiesBuffer[i * 6 + 4] / height;
        enemiesNormalizedBuffer[i * 6 + 5] = enemiesBuffer[i * 6 + 5]; // HP
    }
    inputVector.set(enemiesNormalizedBuffer, k);
    k += enemiesNormalizedBuffer.length;

    // Bullets data
    const bulletsBuffer = TankInputTensor.bulletsData.getBatche(tankEid);
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_BULLETS; i++) {
        if (bulletsBuffer[i * 5] === 0) continue;
        bulletsNormalizedBuffer[i * 5 + 0] = 1;
        bulletsNormalizedBuffer[i * 5 + 1] = ((bulletsBuffer[i * 5 + 1] - tankX) / width);
        bulletsNormalizedBuffer[i * 5 + 2] = ((bulletsBuffer[i * 5 + 2] - tankY) / height);
        bulletsNormalizedBuffer[i * 5 + 3] = bulletsBuffer[i * 5 + 3] / width;
        bulletsNormalizedBuffer[i * 5 + 4] = bulletsBuffer[i * 5 + 4] / height;
    }
    inputVector.set(bulletsNormalizedBuffer, k);
    k += bulletsNormalizedBuffer.length;

    if (inputVector.some(isNaN)) {
        console.error('NaN in inputVector', inputVector);
        debugger;
    }

    return inputVector;
}