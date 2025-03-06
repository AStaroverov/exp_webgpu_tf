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
        position: { x: number, y: number },
        speed: { x: number, y: number },
        turretTarget: Float64Array,
    ) {
        TankInputTensor.health[eid] = health;
        TankInputTensor.position.set(eid, 0, position.x);
        TankInputTensor.position.set(eid, 1, position.y);
        TankInputTensor.speed.set(eid, 0, speed.x);
        TankInputTensor.speed.set(eid, 1, speed.y);
        TankInputTensor.turretTarget.set(eid, 0, turretTarget[0]);
        TankInputTensor.turretTarget.set(eid, 1, turretTarget[1]);
    },

    setEnemiesData(eid: number, index: number, enemyEid: EntityId, hp: number, coord: { x: number, y: number }, speed: {
        x: number,
        y: number
    }) {
        TankInputTensor.enemiesData.set(eid, 6 * index, enemyEid);
        TankInputTensor.enemiesData.set(eid, 6 * index + 1, coord.x);
        TankInputTensor.enemiesData.set(eid, 6 * index + 2, coord.y);
        TankInputTensor.enemiesData.set(eid, 6 * index + 3, speed.x);
        TankInputTensor.enemiesData.set(eid, 6 * index + 4, speed.y);
        TankInputTensor.enemiesData.set(eid, 6 * index + 5, hp);
    },
    resetEnemiesCoords() {
        TankInputTensor.enemiesData.fill(0);
    },
    setBulletsData(eid: number, index: number, bulletEId: EntityId, coord: { x: number, y: number }, speed: {
        x: number,
        y: number
    }) {
        TankInputTensor.bulletsData.set(eid, 4 * index, bulletEId);
        TankInputTensor.bulletsData.set(eid, 4 * index + 1, coord.x);
        TankInputTensor.bulletsData.set(eid, 4 * index + 2, coord.y);
        TankInputTensor.bulletsData.set(eid, 4 * index + 3, speed.x);
        TankInputTensor.bulletsData.set(eid, 4 * index + 4, speed.y);
    },
    resetBulletsCoords() {
        TankInputTensor.bulletsData.fill(0);
    },
});

export function addTankInputTensorComponent(eid: number, { world } = DI) {
    addComponent(world, eid, TankInputTensor);
}
