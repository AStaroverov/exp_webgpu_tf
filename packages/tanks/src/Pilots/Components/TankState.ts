import { addComponent, EntityId, World } from 'bitecs';
import { NestedArray, TypedArray } from 'renderer/src/utils.ts';
import { delegate } from 'renderer/src/delegate.ts';
import { component } from 'renderer/src/ECS/utils.ts';

export const MAX_ENEMIES = 5;
export const ENEMY_BUFFER = 5; // [id, type, hp, x, y]
export const MAX_ALLIES = 5;
export const ALLY_BUFFER = 5; // [id, type, hp, x, y]
export const MAX_BULLETS = 8;
export const BULLET_BUFFER = 5; // [id, x, y, vx, vy]

// Environment rays configuration
export const ENV_RAYS_FORWARD = 5;
export const ENV_RAYS_BACKWARD = 5;
export const ENV_RAYS_LEFT = 1;
export const ENV_RAYS_RIGHT = 1;
export const ENV_RAYS_TOTAL = ENV_RAYS_FORWARD + ENV_RAYS_BACKWARD + ENV_RAYS_LEFT + ENV_RAYS_RIGHT;
export const ENV_RAY_BUFFER = 7; // [rayDirX, rayDirY, hitType, x, y, radius, distance]
export const ENV_RAY_LENGTH = 800;
export const ENV_RAY_SECTOR_ANGLE = Math.PI / 3; // 60 degrees

// Turret rays configuration
export const TURRET_RAY_LENGTH = 1200;
export const TURRET_RAYS_COUNT = 1;
export const TURRET_RAY_BUFFER = 10; // [rayDirX, rayDirY, hitType, x, y, vx, vy, radius, distance, aimingError]
export const TURRET_RAY_ANGLE_OFFSET = Math.PI / 36; // 5 degrees offset for side rays

// Hit types for rays
export const enum RayHitType {
    NONE = 0,
    OBSTACLE = 1,
    ENEMY_VEHICLE = 2,
    ALLY_VEHICLE = 3,
}
export const RAY_HIT_TYPE_COUNT = 4;

export const TankInputTensor = component({
    // Tank
    tankType: TypedArray.f64(delegate.defaultSize),
    health: TypedArray.f64(delegate.defaultSize),
    position: NestedArray.f64(2, delegate.defaultSize),
    rotation: TypedArray.f64(delegate.defaultSize),
    speed: NestedArray.f64(2, delegate.defaultSize),
    turretRotation: TypedArray.f64(delegate.defaultSize),
    colliderRadius: TypedArray.f64(delegate.defaultSize),

    // Enemies [id, hp, x, y]
    enemiesData: NestedArray.f64(ENEMY_BUFFER * MAX_ENEMIES, delegate.defaultSize),

    // Allies [id, hp, x, y]
    alliesData: NestedArray.f64(ALLY_BUFFER * MAX_ALLIES, delegate.defaultSize),

    // Bullets [id, x, y, vx, vy]
    bulletsData: NestedArray.f64(BULLET_BUFFER * MAX_BULLETS, delegate.defaultSize),

    // Environment rays [rayDirX, rayDirY, hitType, x, y, radius, distance] * ENV_RAYS_TOTAL
    envRaysData: NestedArray.f64(ENV_RAY_BUFFER * ENV_RAYS_TOTAL, delegate.defaultSize),

    // Turret rays [rayDirX, rayDirY, hitType, x, y, vx, vy, radius, distance, aimingError] * TURRET_RAYS_COUNT
    turretRaysData: NestedArray.f64(TURRET_RAY_BUFFER * TURRET_RAYS_COUNT, delegate.defaultSize),

    addComponent(world: World, eid: number) {
        addComponent(world, eid, TankInputTensor);
        TankInputTensor.tankType[eid] = 0;
        TankInputTensor.health[eid] = 0;
        TankInputTensor.position.getBatch(eid).fill(0);
        TankInputTensor.rotation[eid] = 0;
        TankInputTensor.speed.getBatch(eid).fill(0);
        TankInputTensor.turretRotation[eid] = 0;
        TankInputTensor.colliderRadius[eid] = 0;
        TankInputTensor.enemiesData.getBatch(eid).fill(0);
        TankInputTensor.alliesData.getBatch(eid).fill(0);
        TankInputTensor.bulletsData.getBatch(eid).fill(0);
        TankInputTensor.envRaysData.getBatch(eid).fill(0);
        TankInputTensor.turretRaysData.getBatch(eid).fill(0);
    },

    // Methods
    setTankData(
        eid: number,
        tankType: number,
        health: number,
        position: Float64Array,
        rotation: number,
        speed: Float64Array,
        turretRotation: number,
        colliderRadius: number,
    ) {
        TankInputTensor.tankType[eid] = tankType;
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
        tankType: number,
        hp: number,
        coord: Float64Array,
    ) {
        const offset = index * ENEMY_BUFFER;
        TankInputTensor.enemiesData.set(eid, offset, enemyEid);
        TankInputTensor.enemiesData.set(eid, offset + 1, tankType);
        TankInputTensor.enemiesData.set(eid, offset + 2, hp);
        TankInputTensor.enemiesData.set(eid, offset + 3, coord[0]);
        TankInputTensor.enemiesData.set(eid, offset + 4, coord[1]);
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
    ) {
        const offset = index * ALLY_BUFFER;
        TankInputTensor.alliesData.set(eid, offset, allyEid);
        TankInputTensor.alliesData.set(eid, offset + 1, tankType);
        TankInputTensor.alliesData.set(eid, offset + 2, hp);
        TankInputTensor.alliesData.set(eid, offset + 3, coord[0]);
        TankInputTensor.alliesData.set(eid, offset + 4, coord[1]);
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

    setEnvRayData(
        eid: number,
        index: number,
        rayDirX: number,
        rayDirY: number,
        hitType: RayHitType,
        x: number,
        y: number,
        radius: number,
        distance: number,
    ) {
        const offset = index * ENV_RAY_BUFFER;
        TankInputTensor.envRaysData.set(eid, offset + 0, rayDirX);
        TankInputTensor.envRaysData.set(eid, offset + 1, rayDirY);
        TankInputTensor.envRaysData.set(eid, offset + 2, hitType);
        TankInputTensor.envRaysData.set(eid, offset + 3, x);
        TankInputTensor.envRaysData.set(eid, offset + 4, y);
        TankInputTensor.envRaysData.set(eid, offset + 5, radius);
        TankInputTensor.envRaysData.set(eid, offset + 6, distance);
    },
    resetEnvRaysData() {
        TankInputTensor.envRaysData.fill(0);
    },

    setTurretRayData(
        eid: number,
        index: number,
        rayDirX: number,
        rayDirY: number,
        hitType: RayHitType,
        x: number,
        y: number,
        vx: number,
        vy: number,
        radius: number,
        distance: number,
        aimingError: number,
    ) {
        const offset = index * TURRET_RAY_BUFFER;
        TankInputTensor.turretRaysData.set(eid, offset + 0, rayDirX);
        TankInputTensor.turretRaysData.set(eid, offset + 1, rayDirY);
        TankInputTensor.turretRaysData.set(eid, offset + 2, hitType);
        TankInputTensor.turretRaysData.set(eid, offset + 3, x);
        TankInputTensor.turretRaysData.set(eid, offset + 4, y);
        TankInputTensor.turretRaysData.set(eid, offset + 5, vx);
        TankInputTensor.turretRaysData.set(eid, offset + 6, vy);
        TankInputTensor.turretRaysData.set(eid, offset + 7, radius);
        TankInputTensor.turretRaysData.set(eid, offset + 8, distance);
        TankInputTensor.turretRaysData.set(eid, offset + 9, aimingError);
    },
    resetTurretRaysData() {
        TankInputTensor.turretRaysData.fill(0);
    },
});
