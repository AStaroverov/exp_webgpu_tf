import { INPUT_DIM } from './consts.ts';
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';

const enemiesNormalizedBuffer = new Float32Array(TANK_INPUT_TENSOR_MAX_ENEMIES * 4);
const bulletsNormalizedBuffer = new Float32Array(TANK_INPUT_TENSOR_MAX_BULLETS * 4);

export function createInputVector(tankEid: number, width: number, height: number, maxSpeed: number) {
    const inputVector = new Float32Array(INPUT_DIM);
    const tankX = TankInputTensor.x[tankEid];
    const tankY = TankInputTensor.y[tankEid];
    let k = 0;

    // Tank state
    inputVector[k++] = TankInputTensor.health[tankEid];
    inputVector[k++] = tankX / width;
    inputVector[k++] = tankY / height;
    inputVector[k++] = TankInputTensor.speed[tankEid] / maxSpeed;
    inputVector[k++] = TankInputTensor.rotation[tankEid] / Math.PI;
    inputVector[k++] = TankInputTensor.turretRotation[tankEid] / Math.PI;
    inputVector[k++] = TankInputTensor.projectileSpeed[tankEid] / maxSpeed;

    // Enemies data
    const enemiesBuffer = TankInputTensor.enemiesData.getBatche(tankEid);
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_ENEMIES; i++) {
        enemiesNormalizedBuffer[i * 4 + 0] = enemiesBuffer[i * 5 + 1] / width;
        enemiesNormalizedBuffer[i * 4 + 1] = enemiesBuffer[i * 5 + 2] / height;
        enemiesNormalizedBuffer[i * 4 + 2] = enemiesBuffer[i * 5 + 3] / maxSpeed;
        enemiesNormalizedBuffer[i * 4 + 3] = enemiesBuffer[i * 5 + 4] / maxSpeed;
    }
    inputVector.set(enemiesNormalizedBuffer, k);
    k += enemiesNormalizedBuffer.length;

    // Bullets data
    const bulletsBuffer = TankInputTensor.bulletsData.getBatche(tankEid);
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_BULLETS; i++) {
        bulletsNormalizedBuffer[i * 4 + 0] = bulletsBuffer[i * 5 + 1] / width;
        bulletsNormalizedBuffer[i * 4 + 1] = bulletsBuffer[i * 5 + 2] / height;
        bulletsNormalizedBuffer[i * 4 + 2] = bulletsBuffer[i * 5 + 3] / maxSpeed;
        bulletsNormalizedBuffer[i * 4 + 3] = bulletsBuffer[i * 5 + 4] / maxSpeed;
    }
    inputVector.set(bulletsNormalizedBuffer, k);
    k += bulletsNormalizedBuffer.length;

    return inputVector;
}