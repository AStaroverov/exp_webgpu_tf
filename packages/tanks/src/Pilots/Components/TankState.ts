import { addComponent, EntityId, World } from 'bitecs';
import { NestedArray, TypedArray } from 'renderer/src/utils.ts';
import { delegate } from 'renderer/src/delegate.ts';
import { component } from 'renderer/src/ECS/utils.ts';

export const MAX_TURRETS = 1;
export const TURRET_BUFFER = 4; // [eid, x, y, rotation]

export const MAX_ENEMIES = 5;
export const ENEMY_BUFFER = 7; // [id, type, hp, x, y, vx, vy]
export const MAX_ALLIES = 5;
export const ALLY_BUFFER = 7; // [id, type, hp, x, y, vx, vy]
export const MAX_BULLETS = 8;
export const BULLET_BUFFER = 5; // [id, x, y, vx, vy]

// Rays configuration
export const RAY_LENGTH = 1200;
export const RAY_BUFFER = 7; // [hitType, hitEid, rootX, rootY, dirX, dirY, distance]

// Total rays count - all rays are unified, targets replace environment rays at their angle
export const RAYS_COUNT = 64;

// Maximum number of direct target rays (enemies + allies)
export const MAX_TARGET_RAYS = MAX_ENEMIES + MAX_ALLIES;

// Hit types for rays
export const enum RayHitType {
    NONE = 0,
    OBSTACLE = 1,
    ENEMY_VEHICLE = 2,
    ALLY_VEHICLE = 3,
}
export const RAY_HIT_TYPE_COUNT = 4;

export const TankInputTensor = component({
    // ---- Tank ----
    tankType: TypedArray.f64(delegate.defaultSize),
    health: TypedArray.f64(delegate.defaultSize),
    position: NestedArray.f64(2, delegate.defaultSize),
    rotation: TypedArray.f64(delegate.defaultSize),
    speed: NestedArray.f64(2, delegate.defaultSize),
    colliderRadius: TypedArray.f64(delegate.defaultSize),
    
    // ---- Rays (environment + turret combined) ----
    raysData: NestedArray.f64(RAY_BUFFER * RAYS_COUNT, delegate.defaultSize),

    // ---- Turrets ----
    turretsData: NestedArray.f64(TURRET_BUFFER * MAX_TURRETS, delegate.defaultSize),

    // ---- Enemies ----
    enemiesData: NestedArray.f64(ENEMY_BUFFER * MAX_ENEMIES, delegate.defaultSize),

    // ---- Allies ----
    alliesData: NestedArray.f64(ALLY_BUFFER * MAX_ALLIES, delegate.defaultSize),

    // ---- Bullets ----
    bulletsData: NestedArray.f64(BULLET_BUFFER * MAX_BULLETS, delegate.defaultSize),


    addComponent(world: World, eid: number) {
        addComponent(world, eid, TankInputTensor);
        TankInputTensor.tankType[eid] = 0;
        TankInputTensor.health[eid] = 0;
        TankInputTensor.position.getBatch(eid).fill(0);
        TankInputTensor.rotation[eid] = 0;
        TankInputTensor.speed.getBatch(eid).fill(0);
        TankInputTensor.colliderRadius[eid] = 0;
        TankInputTensor.raysData.getBatch(eid).fill(0);

        TankInputTensor.turretsData.getBatch(eid).fill(0);

        TankInputTensor.enemiesData.getBatch(eid).fill(0);
        TankInputTensor.alliesData.getBatch(eid).fill(0);
        TankInputTensor.bulletsData.getBatch(eid).fill(0);
    },

    // Methods
    setTankData(
        eid: number,
        tankType: number,
        health: number,
        position: Float64Array,
        rotation: number,
        speed: Float64Array,
        colliderRadius: number,
    ) {
        TankInputTensor.tankType[eid] = tankType;
        TankInputTensor.health[eid] = health;
        TankInputTensor.position.setBatch(eid, position);
        TankInputTensor.rotation[eid] = rotation;
        TankInputTensor.speed.setBatch(eid, speed);
        TankInputTensor.colliderRadius[eid] = colliderRadius;
    },

    setTurretsData(
        eid: number,
        index: number,
        turretEid: EntityId,
        position: Float64Array,
        rotation: number,
    ) {
        const offset = index * TURRET_BUFFER;
        TankInputTensor.turretsData.set(eid, offset + 0, turretEid);
        TankInputTensor.turretsData.set(eid, offset + 1, position[0]);
        TankInputTensor.turretsData.set(eid, offset + 2, position[1]);
        TankInputTensor.turretsData.set(eid, offset + 3, rotation);
    },
    resetTurretsData() {
        TankInputTensor.turretsData.fill(0);
    },

    setEnemiesData(
        eid: number,
        index: number,
        enemyEid: EntityId,
        tankType: number,
        hp: number,
        coord: Float64Array,
        velocity: Float64Array,
    ) {
        const offset = index * ENEMY_BUFFER;
        TankInputTensor.enemiesData.set(eid, offset, enemyEid);
        TankInputTensor.enemiesData.set(eid, offset + 1, tankType);
        TankInputTensor.enemiesData.set(eid, offset + 2, hp);
        TankInputTensor.enemiesData.set(eid, offset + 3, coord[0]);
        TankInputTensor.enemiesData.set(eid, offset + 4, coord[1]);
        TankInputTensor.enemiesData.set(eid, offset + 5, velocity[0]);
        TankInputTensor.enemiesData.set(eid, offset + 6, velocity[1]);
    },
    resetEnemiesCoords() {
        TankInputTensor.enemiesData.fill(0);
    },

    setAlliesData(
        eid: number,
        index: number,
        allyEid: EntityId,
        tankType: number,
        hp: number,
        coord: Float64Array,
        velocity: Float64Array,
    ) {
        const offset = index * ALLY_BUFFER;
        TankInputTensor.alliesData.set(eid, offset, allyEid);
        TankInputTensor.alliesData.set(eid, offset + 1, tankType);
        TankInputTensor.alliesData.set(eid, offset + 2, hp);
        TankInputTensor.alliesData.set(eid, offset + 3, coord[0]);
        TankInputTensor.alliesData.set(eid, offset + 4, coord[1]);
        TankInputTensor.alliesData.set(eid, offset + 5, velocity[0]);
        TankInputTensor.alliesData.set(eid, offset + 6, velocity[1]);
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

    setRayData(
        eid: number,
        index: number,
        hitType: RayHitType,
        hitEid: number,
        rootX: number,
        rootY: number,
        dirX: number,
        dirY: number,
        distance: number,
    ) {
        const offset = index * RAY_BUFFER;
        TankInputTensor.raysData.set(eid, offset + 0, hitType);
        TankInputTensor.raysData.set(eid, offset + 1, hitEid);
        TankInputTensor.raysData.set(eid, offset + 2, rootX);
        TankInputTensor.raysData.set(eid, offset + 3, rootY);
        TankInputTensor.raysData.set(eid, offset + 4, dirX);
        TankInputTensor.raysData.set(eid, offset + 5, dirY);
        TankInputTensor.raysData.set(eid, offset + 6, distance);
    },
    resetRaysData() {
        TankInputTensor.raysData.fill(0);
    },
});
