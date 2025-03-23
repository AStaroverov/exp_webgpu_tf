import { addComponent, EntityId } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { NestedArray, TypedArray } from '../../../../../src/utils.ts';
import { delegate } from '../../../../../src/delegate.ts';
import { component } from '../../../../../src/ECS/utils.ts';

export const TANK_INPUT_TENSOR_MAX_ENEMIES = 4;
export const TANK_INPUT_TENSOR_MAX_BULLETS = 10;
export const TankInputTensor = component({
    health: TypedArray.f64(delegate.defaultSize),
    position: NestedArray.f64(2, delegate.defaultSize),
    speed: NestedArray.f64(2, delegate.defaultSize),
    turretTarget: NestedArray.f64(2, delegate.defaultSize),
    // Bullets -> [id, x,y,vx,vy]
    bulletsData: NestedArray.f64(5 * TANK_INPUT_TENSOR_MAX_BULLETS, delegate.defaultSize),
    // Enemies [id, x,y,vx,vy,ttx,tty]
    enemiesData: NestedArray.f64(7 * TANK_INPUT_TENSOR_MAX_ENEMIES, delegate.defaultSize),

    // Methods
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

    setEnemiesData(eid: number, index: number, enemyEid: EntityId, coord: Float64Array, speed: Float64Array, turretTarget: Float32Array) {
        TankInputTensor.enemiesData.set(eid, 7 * index, enemyEid);
        TankInputTensor.enemiesData.set(eid, 7 * index + 1, coord[0]);
        TankInputTensor.enemiesData.set(eid, 7 * index + 2, coord[1]);
        TankInputTensor.enemiesData.set(eid, 7 * index + 3, speed[0]);
        TankInputTensor.enemiesData.set(eid, 7 * index + 4, speed[1]);
        TankInputTensor.enemiesData.set(eid, 7 * index + 5, turretTarget[0]);
        TankInputTensor.enemiesData.set(eid, 7 * index + 6, turretTarget[1]);
    },
    resetEnemiesCoords() {
        TankInputTensor.enemiesData.fill(0);
    },
    setBulletsData(eid: number, index: number, bulletEId: EntityId, coord: Float64Array, speed: Float64Array) {
        TankInputTensor.bulletsData.set(eid, 5 * index, bulletEId);
        TankInputTensor.bulletsData.set(eid, 5 * index + 1, coord[0]);
        TankInputTensor.bulletsData.set(eid, 5 * index + 2, coord[1]);
        TankInputTensor.bulletsData.set(eid, 5 * index + 3, speed[0]);
        TankInputTensor.bulletsData.set(eid, 5 * index + 4, speed[1]);
    },
    resetBulletsCoords() {
        TankInputTensor.bulletsData.fill(0);
    },
});

export function addTankInputTensorComponent(eid: number, { world } = GameDI) {
    addComponent(world, eid, TankInputTensor);
}
