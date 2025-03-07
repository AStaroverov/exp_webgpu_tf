import { INPUT_DIM } from './consts.ts';
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';

const enemiesNormalizedBuffer = new Float32Array(TANK_INPUT_TENSOR_MAX_ENEMIES * 5);
const bulletsNormalizedBuffer = new Float32Array(TANK_INPUT_TENSOR_MAX_BULLETS * 4);

export function createInputVector(tankEid: number, width: number, height: number) {
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
        enemiesNormalizedBuffer[i * 5 + 0] = ((enemiesBuffer[i * 6 + 1] - tankX) / width);
        enemiesNormalizedBuffer[i * 5 + 1] = ((enemiesBuffer[i * 6 + 2] - tankY) / height);
        enemiesNormalizedBuffer[i * 5 + 2] = enemiesBuffer[i * 6 + 3] / width;
        enemiesNormalizedBuffer[i * 5 + 3] = enemiesBuffer[i * 6 + 4] / height;
        enemiesNormalizedBuffer[i * 5 + 4] = enemiesBuffer[i * 6 + 5]; // HP
    }
    inputVector.set(enemiesNormalizedBuffer, k);
    k += enemiesNormalizedBuffer.length;

    // Bullets data
    const bulletsBuffer = TankInputTensor.bulletsData.getBatche(tankEid);
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_BULLETS; i++) {
        bulletsNormalizedBuffer[i * 4 + 0] = ((bulletsBuffer[i * 5 + 1] - tankX) / width);
        bulletsNormalizedBuffer[i * 4 + 1] = ((bulletsBuffer[i * 5 + 2] - tankY) / height);
        bulletsNormalizedBuffer[i * 4 + 2] = bulletsBuffer[i * 5 + 3] / width;
        bulletsNormalizedBuffer[i * 4 + 3] = bulletsBuffer[i * 5 + 4] / height;
    }
    inputVector.set(bulletsNormalizedBuffer, k);
    k += bulletsNormalizedBuffer.length;

    return inputVector;
}