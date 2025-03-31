import { addComponent, EntityId } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { NestedArray, TypedArray } from '../../../../../src/utils.ts';
import { delegate } from '../../../../../src/delegate.ts';
import { component } from '../../../../../src/ECS/utils.ts';

export const TANK_INPUT_TENSOR_MAX_ENEMIES = 3;
export const TANK_INPUT_TENSOR_ENEMY_BUFFER = 7;
export const TANK_INPUT_TENSOR_MAX_BULLETS = 20;
export const TANK_INPUT_TENSOR_BULLET_BUFFER = 5;
export const TankInputTensor = component({
    health: TypedArray.f64(delegate.defaultSize),
    position: NestedArray.f64(2, delegate.defaultSize),
    speed: NestedArray.f64(2, delegate.defaultSize),
    turretTarget: NestedArray.f64(2, delegate.defaultSize),
    // Bullets -> [id, x,y,vx,vy]
    bulletsData: NestedArray.f64(TANK_INPUT_TENSOR_BULLET_BUFFER * TANK_INPUT_TENSOR_MAX_BULLETS, delegate.defaultSize),
    // Enemies [id, x,y,vx,vy,ttx,tty]
    enemiesData: NestedArray.f64(TANK_INPUT_TENSOR_ENEMY_BUFFER * TANK_INPUT_TENSOR_MAX_ENEMIES, delegate.defaultSize),

    // Methods
    setEnemiesData(eid: number, index: number, enemyEid: EntityId, coord: Float64Array, speed: Float64Array, turretTarget: Float32Array) {
        const b = TANK_INPUT_TENSOR_ENEMY_BUFFER;
        TankInputTensor.enemiesData.set(eid, b * index, enemyEid);
        TankInputTensor.enemiesData.set(eid, b * index + 1, coord[0]);
        TankInputTensor.enemiesData.set(eid, b * index + 2, coord[1]);
        TankInputTensor.enemiesData.set(eid, b * index + 3, speed[0]);
        TankInputTensor.enemiesData.set(eid, b * index + 4, speed[1]);
        TankInputTensor.enemiesData.set(eid, b * index + 5, turretTarget[0]);
        TankInputTensor.enemiesData.set(eid, b * index + 6, turretTarget[1]);
    },
    setTankData(
        eid: number,
        health: number,
        speed: Float64Array,
        position: Float64Array,
        turretTarget: Float32Array,
    ) {
        TankInputTensor.health[eid] = health;
        TankInputTensor.speed.setBatch(eid, speed);
        TankInputTensor.position.setBatch(eid, position);
        TankInputTensor.turretTarget.setBatch(eid, turretTarget);
    },

    resetEnemiesCoords() {
        TankInputTensor.enemiesData.fill(0);
    },
    setBulletsData(eid: number, index: number, bulletEId: EntityId, coord: Float64Array, speed: Float64Array) {
        const b = TANK_INPUT_TENSOR_BULLET_BUFFER;
        TankInputTensor.bulletsData.set(eid, b * index, bulletEId);
        TankInputTensor.bulletsData.set(eid, b * index + 1, coord[0]);
        TankInputTensor.bulletsData.set(eid, b * index + 2, coord[1]);
        TankInputTensor.bulletsData.set(eid, b * index + 3, speed[0]);
        TankInputTensor.bulletsData.set(eid, b * index + 4, speed[1]);
    },
    resetBulletsCoords() {
        TankInputTensor.bulletsData.fill(0);
    },
});

export function addTankInputTensorComponent(eid: number, { world } = GameDI) {
    addComponent(world, eid, TankInputTensor);
}
