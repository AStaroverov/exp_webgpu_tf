import { addComponent } from 'bitecs';
import { DI } from '../../DI';
import { NestedArray, TypedArray } from '../../../../../src/utils.ts';
import { delegate } from '../../../../../src/delegate.ts';

export const TANK_INPUT_TENSOR_MAX_ENEMIES = 4;
export const TankInputTensor = ({
    health: TypedArray.f64(delegate.defaultSize),
    x: TypedArray.f64(delegate.defaultSize),
    y: TypedArray.f64(delegate.defaultSize),
    speed: TypedArray.f64(delegate.defaultSize),
    rotation: TypedArray.f64(delegate.defaultSize),
    turretRotation: TypedArray.f64(delegate.defaultSize),
    projectileSpeed: TypedArray.f64(delegate.defaultSize),
    // Enemies
    enemiesX: NestedArray.f64(TANK_INPUT_TENSOR_MAX_ENEMIES, delegate.defaultSize),
    enemiesY: NestedArray.f64(TANK_INPUT_TENSOR_MAX_ENEMIES, delegate.defaultSize),
    enemiesSpeed: NestedArray.f64(TANK_INPUT_TENSOR_MAX_ENEMIES, delegate.defaultSize),
    enemiesRotation: NestedArray.f64(TANK_INPUT_TENSOR_MAX_ENEMIES, delegate.defaultSize),
    enemiesTurretRotation: NestedArray.f64(TANK_INPUT_TENSOR_MAX_ENEMIES, delegate.defaultSize),
});

export function addTankInputTensorComponent(eid: number, { world } = DI) {
    addComponent(world, eid, TankInputTensor);
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
    TankInputTensor.enemiesX.set(eid, index, x);
    TankInputTensor.enemiesY.set(eid, index, y);
    TankInputTensor.enemiesSpeed.set(eid, index, speed);
    TankInputTensor.enemiesRotation.set(eid, index, rotation);
    TankInputTensor.enemiesTurretRotation.set(eid, index, turretRotation);
}

export function resetTankInputTensorEnemy(eid: number, index: number) {
    TankInputTensor.enemiesX.set(eid, index, 0);
    TankInputTensor.enemiesY.set(eid, index, 0);
    TankInputTensor.enemiesSpeed.set(eid, index, 0);
    TankInputTensor.enemiesRotation.set(eid, index, 0);
    TankInputTensor.enemiesTurretRotation.set(eid, index, 0);
}