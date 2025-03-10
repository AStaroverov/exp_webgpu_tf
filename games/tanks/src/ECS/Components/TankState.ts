import { addComponent, EntityId } from 'bitecs';
import { DI } from '../../DI';
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
    // Enemies [id, x,y,vx,vy,hp]
    enemiesData: NestedArray.f64(6 * TANK_INPUT_TENSOR_MAX_ENEMIES, delegate.defaultSize),

    // Methods
    setTankData(
        eid: number,
        health: number,
        position: Float64Array,
        speed: Float64Array,
        turretTarget: Float64Array,
    ) {
        TankInputTensor.health[eid] = health;
        TankInputTensor.position.setBatch(eid, position);
        TankInputTensor.speed.setBatch(eid, speed);
        TankInputTensor.turretTarget.set(eid, 0, turretTarget[0]);
        TankInputTensor.turretTarget.set(eid, 1, turretTarget[1]);
    },

    setEnemiesData(eid: number, index: number, enemyEid: EntityId, hp: number, coord: Float64Array, speed: Float64Array) {
        TankInputTensor.enemiesData.set(eid, 6 * index, enemyEid);
        TankInputTensor.enemiesData.set(eid, 6 * index + 1, coord[0]);
        TankInputTensor.enemiesData.set(eid, 6 * index + 2, coord[1]);
        TankInputTensor.enemiesData.set(eid, 6 * index + 3, speed[0]);
        TankInputTensor.enemiesData.set(eid, 6 * index + 4, speed[1]);
        TankInputTensor.enemiesData.set(eid, 6 * index + 5, hp);
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

export function addTankInputTensorComponent(eid: number, { world } = DI) {
    addComponent(world, eid, TankInputTensor);
}
