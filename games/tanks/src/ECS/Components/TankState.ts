import { addComponent, defineComponent, Types } from 'bitecs';
import { DI } from '../../DI';

export const TANK_INPUT_TENSOR_MAX_ENEMIES = 4;
export const TankInputTensor = defineComponent({
    health: Types.f64,
    x: Types.f64,
    y: Types.f64,
    speed: Types.f64,
    rotation: Types.f64,
    turretRotation: Types.f64,
    projectileSpeed: Types.f64,
    // Enemies
    enemiesX: [Types.f64, TANK_INPUT_TENSOR_MAX_ENEMIES],
    enemiesY: [Types.f64, TANK_INPUT_TENSOR_MAX_ENEMIES],
    enemiesSpeed: [Types.f64, TANK_INPUT_TENSOR_MAX_ENEMIES],
    enemiesRotation: [Types.f64, TANK_INPUT_TENSOR_MAX_ENEMIES],
    enemiesTurretRotation: [Types.f64, TANK_INPUT_TENSOR_MAX_ENEMIES],
});

export function addTankInputTensorComponent(eid: number, { world } = DI) {
    addComponent(world, TankInputTensor, eid);
}

export function setTankInputTensorSelf(eid: number, health: number, x: number, y: number, speed: number, rotation: number, turretRotation: number, projectileSpeed: number) {
    TankInputTensor.health[eid] = health;
    TankInputTensor.x[eid] = x;
    TankInputTensor.y[eid] = y;
    TankInputTensor.speed[eid] = speed;
    TankInputTensor.rotation[eid] = rotation;
    TankInputTensor.turretRotation[eid] = turretRotation;
    TankInputTensor.projectileSpeed[eid] = projectileSpeed;
}

export function setTankInputTensorEnemy(eid: number, index: number, x: number, y: number, speed: number, rotation: number, turretRotation: number) {
    TankInputTensor.enemiesX[eid][index] = x;
    TankInputTensor.enemiesY[eid][index] = y;
    TankInputTensor.enemiesSpeed[eid][index] = speed;
    TankInputTensor.enemiesRotation[eid][index] = rotation;
    TankInputTensor.enemiesTurretRotation[eid][index] = turretRotation;
}

export function resetTankInputTensorEnemy(eid: number, index: number) {
    TankInputTensor.enemiesX[eid][index] = 0;
    TankInputTensor.enemiesY[eid][index] = 0;
    TankInputTensor.enemiesSpeed[eid][index] = 0;
    TankInputTensor.enemiesRotation[eid][index] = 0;
    TankInputTensor.enemiesTurretRotation[eid][index] = 0;
}