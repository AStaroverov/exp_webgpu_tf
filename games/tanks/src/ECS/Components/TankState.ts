import { addComponent } from 'bitecs';
import { DI } from '../../DI';
import { component, NestedArray, TypedArray } from '../../../../../src/utils.ts';
import { delegate } from '../../../../../src/delegate.ts';

export const TANK_INPUT_TENSOR_MAX_ENEMIES = 4;
export const TANK_INPUT_TENSOR_MAX_BULLETS = 10;
export const TankInputTensor = component({
    health: TypedArray.f64(delegate.defaultSize),
    x: TypedArray.f64(delegate.defaultSize),
    y: TypedArray.f64(delegate.defaultSize),
    speed: TypedArray.f64(delegate.defaultSize),
    rotation: TypedArray.f64(delegate.defaultSize),
    turretRotation: TypedArray.f64(delegate.defaultSize),
    projectileSpeed: TypedArray.f64(delegate.defaultSize),
    // Bullets id -> [x,y,vx,vy][10]
    bulletsData: NestedArray.f64(4 * TANK_INPUT_TENSOR_MAX_BULLETS, delegate.defaultSize),
    // Enemies [x,y,vx,vy]
    enemiesData: NestedArray.f64(4 * TANK_INPUT_TENSOR_MAX_ENEMIES, delegate.defaultSize),

    // Methods
    setEnemiesData(eid: number, index: number, coord: { x: number, y: number }, speed: { x: number, y: number }) {
        TankInputTensor.enemiesData.set(eid, 4 * index, coord.x);
        TankInputTensor.enemiesData.set(eid, 4 * index + 1, coord.y);
        TankInputTensor.enemiesData.set(eid, 4 * index + 2, speed.x);
        TankInputTensor.enemiesData.set(eid, 4 * index + 3, speed.y);
    },
    resetEnemiesCoords() {
        TankInputTensor.enemiesData.fill(0);
    },

    setBulletsData(eid: number, index: number, coord: { x: number, y: number }, speed: { x: number, y: number }) {
        TankInputTensor.bulletsData.set(eid, 4 * index, coord.x);
        TankInputTensor.bulletsData.set(eid, 4 * index + 1, coord.y);
        TankInputTensor.bulletsData.set(eid, 4 * index + 2, speed.x);
        TankInputTensor.bulletsData.set(eid, 4 * index + 3, speed.y);
    },
    resetBulletsCoords() {
        TankInputTensor.bulletsData.fill(0);
    },

});

export function addTankInputTensorComponent(eid: number, { world } = DI) {
    addComponent(world, eid, TankInputTensor);
}

export function setTankInputTensorSelf(
    eid: number,
    health: number,
    x: number,
    y: number,
    speed: number,
    rotation: number,
    turretRotation: number,
    projectileSpeed: number,
) {
    TankInputTensor.health[eid] = health;
    TankInputTensor.x[eid] = x;
    TankInputTensor.y[eid] = y;
    TankInputTensor.speed[eid] = speed;
    TankInputTensor.rotation[eid] = rotation;
    TankInputTensor.turretRotation[eid] = turretRotation;
    TankInputTensor.projectileSpeed[eid] = projectileSpeed;
}