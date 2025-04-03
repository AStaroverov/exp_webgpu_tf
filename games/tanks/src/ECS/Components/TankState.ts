import { addComponent, EntityId } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { NestedArray, TypedArray } from '../../../../../src/utils.ts';
import { delegate } from '../../../../../src/delegate.ts';
import { component } from '../../../../../src/ECS/utils.ts';

export const MAX_ENEMIES = 3;
export const ENEMY_BUFFER = 7;
export const MAX_BULLETS = 10;
export const BULLET_BUFFER = 5;
export const TankInputTensor = component({
    health: TypedArray.f64(delegate.defaultSize),
    position: NestedArray.f64(2, delegate.defaultSize),
    speed: NestedArray.f64(2, delegate.defaultSize),
    turretTarget: NestedArray.f64(2, delegate.defaultSize),
    // Bullets -> [id, x,y,vx,vy]
    bulletsData: NestedArray.f64(BULLET_BUFFER * MAX_BULLETS, delegate.defaultSize),
    // Enemies [id, x,y,vx,vy,ttx,tty]
    enemiesData: NestedArray.f64(ENEMY_BUFFER * MAX_ENEMIES, delegate.defaultSize),

    // Methods
    setEnemiesData(eid: number, index: number, enemyEid: EntityId, coord: Float64Array, speed: Float64Array, turretTarget: Float32Array) {
        const s = ENEMY_BUFFER;
        TankInputTensor.enemiesData.set(eid, s * index, enemyEid);
        TankInputTensor.enemiesData.set(eid, s * index + 1, coord[0]);
        TankInputTensor.enemiesData.set(eid, s * index + 2, coord[1]);
        TankInputTensor.enemiesData.set(eid, s * index + 3, speed[0]);
        TankInputTensor.enemiesData.set(eid, s * index + 4, speed[1]);
        TankInputTensor.enemiesData.set(eid, s * index + 5, turretTarget[0]);
        TankInputTensor.enemiesData.set(eid, s * index + 6, turretTarget[1]);
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
        const s = BULLET_BUFFER;
        TankInputTensor.bulletsData.set(eid, s * index, bulletEId);
        TankInputTensor.bulletsData.set(eid, s * index + 1, coord[0]);
        TankInputTensor.bulletsData.set(eid, s * index + 2, coord[1]);
        TankInputTensor.bulletsData.set(eid, s * index + 3, speed[0]);
        TankInputTensor.bulletsData.set(eid, s * index + 4, speed[1]);
    },
    resetBulletsCoords() {
        TankInputTensor.bulletsData.fill(0);
    },
});

export function addTankInputTensorComponent(eid: number, { world } = GameDI) {
    addComponent(world, eid, TankInputTensor);
}
