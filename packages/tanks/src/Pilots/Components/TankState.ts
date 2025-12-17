import { addComponent, EntityId, World } from 'bitecs';
import { NestedArray, TypedArray } from 'renderer/src/utils.ts';
import { delegate } from 'renderer/src/delegate.ts';
import { component } from 'renderer/src/ECS/utils.ts';

export const MAX_ENEMIES = 3;
export const ENEMY_BUFFER = 9;
export const MAX_ALLIES = 3;
export const ALLY_BUFFER = 9;
export const MAX_BULLETS = 8;
export const BULLET_BUFFER = 5;
export const TankInputTensor = component({
    // Tank
    health: TypedArray.f64(delegate.defaultSize),
    position: NestedArray.f64(2, delegate.defaultSize),
    rotation: TypedArray.f64(delegate.defaultSize),
    speed: NestedArray.f64(2, delegate.defaultSize),
    turretRotation: TypedArray.f64(delegate.defaultSize),
    colliderRadius: TypedArray.f64(delegate.defaultSize),

    // Enemies [id, hp,x,y,r,vx,vy,tr,collider]
    enemiesData: NestedArray.f64(ENEMY_BUFFER * MAX_ENEMIES, delegate.defaultSize),

    // Allies [id, hp,x,y,r,vx,vy,tr,collider]
    alliesData: NestedArray.f64(ALLY_BUFFER * MAX_ALLIES, delegate.defaultSize),

    // Bullets [id, x,y,vx,vy]
    bulletsData: NestedArray.f64(BULLET_BUFFER * MAX_BULLETS, delegate.defaultSize),

    addComponent(world: World, eid: number) {
        addComponent(world, eid, TankInputTensor);
        TankInputTensor.health[eid] = 0;
        TankInputTensor.position.getBatch(eid).fill(0);
        TankInputTensor.rotation[eid] = 0;
        TankInputTensor.speed.getBatch(eid).fill(0);
        TankInputTensor.turretRotation[eid] = 0;
        TankInputTensor.colliderRadius[eid] = 0;
        TankInputTensor.enemiesData.getBatch(eid).fill(0);
        TankInputTensor.alliesData.getBatch(eid).fill(0);
        TankInputTensor.bulletsData.getBatch(eid).fill(0);
    },

    // Methods
    setTankData(
        eid: number,
        health: number,
        position: Float64Array,
        rotation: number,
        speed: Float64Array,
        turretRotation: number,
        colliderRadius: number,
    ) {
        TankInputTensor.health[eid] = health;
        TankInputTensor.position.setBatch(eid, position);
        TankInputTensor.rotation[eid] = rotation;
        TankInputTensor.speed.setBatch(eid, speed);
        TankInputTensor.turretRotation[eid] = turretRotation;
        TankInputTensor.colliderRadius[eid] = colliderRadius;
    },

    setEnemiesData(
        eid: number,
        index: number,
        enemyEid: EntityId,
        hp: number,
        coord: Float64Array,
        rotation: number,
        speed: Float64Array,
        turretRotation: number,
        colliderRadius: number,
    ) {
        const offset = index * ENEMY_BUFFER;
        TankInputTensor.enemiesData.set(eid, offset, enemyEid);
        TankInputTensor.enemiesData.set(eid, offset + 1, hp);
        TankInputTensor.enemiesData.set(eid, offset + 2, coord[0]);
        TankInputTensor.enemiesData.set(eid, offset + 3, coord[1]);
        TankInputTensor.enemiesData.set(eid, offset + 4, rotation);
        TankInputTensor.enemiesData.set(eid, offset + 5, speed[0]);
        TankInputTensor.enemiesData.set(eid, offset + 6, speed[1]);
        TankInputTensor.enemiesData.set(eid, offset + 7, turretRotation);
        TankInputTensor.enemiesData.set(eid, offset + 8, colliderRadius);
    },
    resetEnemiesCoords() {
        TankInputTensor.enemiesData.fill(0);
    },

    setAlliesData(
        eid: number,
        index: number,
        allyEid: EntityId,
        hp: number,
        coord: Float64Array,
        rotation: number,
        speed: Float64Array,
        turretRotation: number,
        colliderRadius: number,
    ) {
        const offset = index * ALLY_BUFFER;
        TankInputTensor.alliesData.set(eid, offset, allyEid);
        TankInputTensor.alliesData.set(eid, offset + 1, hp);
        TankInputTensor.alliesData.set(eid, offset + 2, coord[0]);
        TankInputTensor.alliesData.set(eid, offset + 3, coord[1]);
        TankInputTensor.alliesData.set(eid, offset + 4, rotation);
        TankInputTensor.alliesData.set(eid, offset + 5, speed[0]);
        TankInputTensor.alliesData.set(eid, offset + 6, speed[1]);
        TankInputTensor.alliesData.set(eid, offset + 7, turretRotation);
        TankInputTensor.alliesData.set(eid, offset + 8, colliderRadius);
    },
    resetAlliesCoords() {
        TankInputTensor.alliesData.fill(0);
    },

    setBulletsData(eid: number, index: number, bulletEId: EntityId, coord: Float64Array, speed: Float64Array) {
        const offset = index * BULLET_BUFFER;
        TankInputTensor.bulletsData.set(eid, offset, bulletEId);
        TankInputTensor.bulletsData.set(eid, offset + 1, coord[0]);
        TankInputTensor.bulletsData.set(eid, offset + 2, coord[1]);
        TankInputTensor.bulletsData.set(eid, offset + 3, speed[0]);
        TankInputTensor.bulletsData.set(eid, offset + 4, speed[1]);
    },
    resetBulletsCoords() {
        TankInputTensor.bulletsData.fill(0);
    },
});
