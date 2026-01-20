import { Ball, Collider, Ray, Vector2 } from '@dimforge/rapier2d-simd';
import { EntityId, hasComponent, query } from 'bitecs';
import { cos, hypot, sin } from '../../../../../../lib/math.ts';
import { shuffle } from '../../../../../../lib/shuffle.ts';
import { GameDI } from '../../../GameEngine/DI/GameDI.js';
import { getEntityIdByPhysicalId, RigidBodyRef, RigidBodyState } from '../../../GameEngine/ECS/Components/Physical.js';
import { PlayerRef } from '../../../GameEngine/ECS/Components/PlayerRef.js';
import { Tank } from '../../../GameEngine/ECS/Components/Tank.js';
import { Vehicle } from '../../../GameEngine/ECS/Components/Vehicle.js';
import { TeamRef } from '../../../GameEngine/ECS/Components/TeamRef.js';
import { GameMap } from '../../../GameEngine/ECS/Entities/GameMap.js';
import { CollisionGroup, createCollisionGroups } from '../../../GameEngine/Physical/createRigid.js';
import { hasIntersectionVectorAndCircle } from '../../../GameEngine/Utils/intersections.js';
import {
    MAX_ALLIES,
    MAX_BULLETS,
    MAX_ENEMIES,
    TankInputTensor,
    RAYS_COUNT,
    RAY_LENGTH,
    RayHitType
} from '../Components/TankState.js';

import { HeuristicsData } from '../../../GameEngine/ECS/Components/HeuristicsData.js';
import { getTankHealth } from '../../../GameEngine/ECS/Entities/Tank/TankUtils.js';
import { Parent } from '../../../GameEngine/ECS/Components/Parent.js';
import { ALL_VEHICLE_PARTS_MASK, CollisionGroupConfig } from '../../../GameEngine/Config/physics.js';

// Temp arrays for offset-adjusted positions
const tempTankPosition = new Float64Array(2);
const tempTurretPosition = new Float64Array(2);
const tempEnemyPosition = new Float64Array(2);
const tempAllyPosition = new Float64Array(2);
const tempBulletPosition = new Float64Array(2);


// Reusable ray objects
const rayOrigin = new Vector2(0, 0);
const rayDir = new Vector2(0, 0);

// Collision mask for environment rays: obstacles + vehicle parts
const ENV_RAY_COLLISION_MASK = CollisionGroupConfig.OBSTACLE | ALL_VEHICLE_PARTS_MASK;

export function snapshotTankInputTensor({ world } = GameDI) {
    TankInputTensor.resetRaysData();
    TankInputTensor.resetTurretsData();
    TankInputTensor.resetEnemiesCoords();
    TankInputTensor.resetAlliesCoords();
    TankInputTensor.resetBulletsCoords();
    
    const vehicleEids = query(world, [Vehicle, TankInputTensor, RigidBodyState]);

    for (let i = 0; i < vehicleEids.length; i++) {
        const vehicleEid = vehicleEids[i];
        const turretEid = Tank.turretEId[vehicleEid];
        const myTeamId = TeamRef.id[vehicleEid];

        // Set vehicle data (with offset-adjusted position)
        const health = getTankHealth(vehicleEid);
        const position = RigidBodyState.position.getBatch(vehicleEid);
        const rotation = RigidBodyState.rotation[vehicleEid];
        const linvel = RigidBodyState.linvel.getBatch(vehicleEid);
        const approximateColliderRadius = HeuristicsData.approxColliderRadius[vehicleEid];
        const tankType = Vehicle.type[vehicleEid];
        
        tempTankPosition[0] = position[0] - GameMap.offsetX;
        tempTankPosition[1] = position[1] - GameMap.offsetY;
        TankInputTensor.setTankData(
            vehicleEid,
            tankType,
            health,
            tempTankPosition,
            rotation,
            linvel,
            approximateColliderRadius,
        );
        
        const turretPosition = RigidBodyState.position.getBatch(turretEid);
        const turretRotation = RigidBodyState.rotation[turretEid];

        tempTurretPosition[0] = turretPosition[0] - GameMap.offsetX;
        tempTurretPosition[1] = turretPosition[1] - GameMap.offsetY;
        TankInputTensor.setTurretsData(
            vehicleEid,
            0,
            turretEid,
            tempTurretPosition,
            turretRotation,
        );

        // Find closest enemies
        const enemiesEids = findTankEnemiesEids(vehicleEid);

        for (let j = 0; j < enemiesEids.length; j++) {
            const enemyEid = enemiesEids[j];
            const enemyPosition = RigidBodyState.position.getBatch(enemyEid);
            tempEnemyPosition[0] = enemyPosition[0] - GameMap.offsetX;
            tempEnemyPosition[1] = enemyPosition[1] - GameMap.offsetY;

            const enemyTurretEid = Tank.turretEId[enemyEid];
            const enemyTurretRotation = RigidBodyState.rotation[enemyTurretEid];
            const enemyColliderRadius = HeuristicsData.approxColliderRadius[enemyEid];

            TankInputTensor.setEnemiesData(
                vehicleEid,
                j,
                enemyEid,
                Vehicle.type[enemyEid],
                getTankHealth(enemyEid),
                tempEnemyPosition,
                RigidBodyState.linvel.getBatch(enemyEid),
                enemyTurretRotation,
                enemyColliderRadius,
            );
        }

        // Find closest allies
        const alliesEids = findTankAlliesEids(vehicleEid);

        for (let j = 0; j < alliesEids.length; j++) {
            const allyEid = alliesEids[j];
            const allyPosition = RigidBodyState.position.getBatch(allyEid);
            tempAllyPosition[0] = allyPosition[0] - GameMap.offsetX;
            tempAllyPosition[1] = allyPosition[1] - GameMap.offsetY;

            const allyTurretEid = Tank.turretEId[allyEid];
            const allyTurretRotation = RigidBodyState.rotation[allyTurretEid];
            const allyColliderRadius = HeuristicsData.approxColliderRadius[allyEid];

            TankInputTensor.setAlliesData(
                vehicleEid,
                j,
                allyEid,
                Vehicle.type[allyEid],
                getTankHealth(allyEid),
                tempAllyPosition,
                RigidBodyState.linvel.getBatch(allyEid),
                allyTurretRotation,
                allyColliderRadius,
            );
        }

        // Cast unified rays - environment rays with direct rays to targets integrated
        castUnifiedRays(
            vehicleEid,
            myTeamId,
            position[0],
            position[1],
            rotation,
            [...enemiesEids, ...alliesEids],
        );

        // Find closest bullets
        const bulletsEids = findTankDangerBullets(vehicleEid);

        for (let j = 0; j < bulletsEids.length; j++) {
            const bulletEid = bulletsEids[j];
            const bulletPosition = RigidBodyState.position.getBatch(bulletEid);
            tempBulletPosition[0] = bulletPosition[0] - GameMap.offsetX;
            tempBulletPosition[1] = bulletPosition[1] - GameMap.offsetY;

            TankInputTensor.setBulletsData(
                vehicleEid,
                j,
                bulletEid,
                tempBulletPosition,
                RigidBodyState.linvel.getBatch(bulletEid),
            );
        }
    }
}

/**
 * Creates a filter predicate that excludes colliders belonging to the specified vehicle
 */
function createExcludeOwnVehiclePredicate(
    vehicleEid: EntityId,
): (collider: Collider) => boolean {
    return (collider: Collider): boolean => {
        const rigidBody = collider.parent();
        if (!rigidBody) return true;
        
        const eid = getEntityIdByPhysicalId(rigidBody.handle);
        if (eid === 0) return true;
        
        // Check if this entity is the vehicle itself
        if (eid === vehicleEid) return false;
        
        // Check if this entity belongs to the vehicle (via Parent chain)
        const ownerVehicleEid = findVehicleFromPart(eid);
        if (ownerVehicleEid === vehicleEid) return false;
        
        return true;
    };
}

/**
 * Angle step for unified ray system (64 rays covering 360 degrees)
 */
const UNIFIED_ANGLE_STEP = (2 * Math.PI) / RAYS_COUNT;

/**
 * Pre-allocated array for ray angles override.
 * NaN means no override (use default environment ray angle).
 */
const targetRayAngles = new Float64Array(RAYS_COUNT);

/**
 * Reset target ray angles to default (evenly distributed)
 */
function resetTargetRayAngles() {
    for (let i = 0; i < RAYS_COUNT; i++) {
        targetRayAngles[i] = i * UNIFIED_ANGLE_STEP;
    }
}

/**
 * Cast unified rays - 64 rays evenly distributed around the agent.
 * For each target (enemy/ally), the ray at the closest angle slot is overridden
 * to point directly at that target. All rays use standard RAY_LENGTH.
 */
function castUnifiedRays(
    parentEid: EntityId,
    myTeamId: number,
    posX: number,
    posY: number,
    rotation: number,
    targetEids: EntityId[],
    { physicalWorld } = GameDI,
) {
    rayOrigin.x = posX;
    rayOrigin.y = posY;

    const filterPredicate = createExcludeOwnVehiclePredicate(parentEid);

    // Reset and populate target ray angles
    resetTargetRayAngles();
    
    for (let i = 0; i < targetEids.length; i++) {
        const targetEid = targetEids[i];
        const targetPosition = RigidBodyState.position.getBatch(targetEid);
        
        const dx = targetPosition[0] - posX;
        const dy = targetPosition[1] - posY;
        const angleToTarget = Math.atan2(dy, dx);
        
        // Find which ray slot this angle corresponds to (relative to forward direction)
        let relativeAngle = angleToTarget - rotation;
        while (relativeAngle < 0) relativeAngle += 2 * Math.PI;
        while (relativeAngle >= 2 * Math.PI) relativeAngle -= 2 * Math.PI;
        
        const rayIndex = Math.round(relativeAngle / UNIFIED_ANGLE_STEP) % RAYS_COUNT;
        targetRayAngles[rayIndex] = relativeAngle;
    }

    // Cast all 64 rays
    for (let i = 0; i < RAYS_COUNT; i++) {
        const angle = rotation + targetRayAngles[i];
        
        rayDir.x = cos(angle);
        rayDir.y = sin(angle);

        const ray = new Ray(rayOrigin, rayDir);
        const hit = physicalWorld.castRay(
            ray,
            RAY_LENGTH,
            true,
            undefined,
            createCollisionGroups(CollisionGroup.ALL, ENV_RAY_COLLISION_MASK),
            undefined,
            undefined,
            filterPredicate,
        );

        const hitEid = hit ? getEntityIdByPhysicalId(hit.collider.handle) : 0;
        const rayHitType = hit && getRayHitType(hitEid, myTeamId);
        const distance = hit?.timeOfImpact ?? RAY_LENGTH;
        
        TankInputTensor.setRayData(
            parentEid,
            i,
            rayHitType ?? RayHitType.NONE,
            hitEid,
            rayOrigin.x - GameMap.offsetX,
            rayOrigin.y - GameMap.offsetY,
            rayDir.x,
            rayDir.y,
            distance,
        );
    }
}

function getRayHitType(eid: EntityId, myTeamId: number): RayHitType {
    const collisionGroup = getCollisionGroupFromEntity(eid);

    if (collisionGroup === 0) {
        return RayHitType.NONE;
    }

    if (collisionGroup & ALL_VEHICLE_PARTS_MASK) {
        // It's a vehicle part - traverse parent to get vehicle
        const vehicleEid = findVehicleFromPart(eid);
        if (vehicleEid === 0) return RayHitType.NONE;

        const hitTeamId = TeamRef.id[vehicleEid];
        return hitTeamId === myTeamId ? RayHitType.ALLY_VEHICLE : RayHitType.ENEMY_VEHICLE;
    }
    
    if (collisionGroup & CollisionGroupConfig.OBSTACLE) {
        return RayHitType.OBSTACLE;
    }

    return RayHitType.NONE;
}

/**
 * Get collision group from entity's collider
 */
function getCollisionGroupFromEntity(eid: EntityId, { physicalWorld } = GameDI): number {
    const rigidBodyHandle = RigidBodyRef.id[eid];
    if (rigidBodyHandle === 0) return 0;
    
    const rigidBody = physicalWorld.getRigidBody(rigidBodyHandle);
    if (!rigidBody) return 0;
    
    const collider = rigidBody.collider(0);
    if (!collider) return 0;
    
    // Get collision groups - the membership is in the upper 16 bits
    const groups = collider.collisionGroups();
    return (groups >> 16) & 0xFFFF;
}

/**
 * Traverse parent hierarchy to find vehicle entity
 */
export function findVehicleFromPart(partEid: EntityId, { world } = GameDI): EntityId {
    let currentEid = partEid;
    const maxDepth = 5; // Prevent infinite loops, should 2 enough for most cases
    
    for (let i = 0; i < maxDepth; i++) {
        if (hasComponent(world, currentEid, Vehicle)) {
            return currentEid;
        }
        
        const parentId = Parent.id[currentEid];
        if (parentId === 0 || parentId === currentEid) {
            break;
        }
        currentEid = parentId;
    }
    
    return 0;
}

export function findTankEnemiesEids(tankEid: EntityId) {
    const tankTeamId = TeamRef.id[tankEid];
    return findAllTankEids(tankEid, MAX_ENEMIES, (eid) => TeamRef.id[eid] !== tankTeamId);
}

export function findTankAlliesEids(tankEid: EntityId) {
    const tankTeamId = TeamRef.id[tankEid];
    return findAllTankEids(tankEid, MAX_ALLIES, (eid) => TeamRef.id[eid] === tankTeamId);
}

function findAllTankEids(
    tankEid: EntityId, 
    limit: number, 
    select: (eid: EntityId) => boolean, 
    { world } = GameDI,
): EntityId[] {
    const allVehicles = query(world, [Vehicle, TankInputTensor]);
    const result: EntityId[] = [];
    
    for (let i = 0; i < allVehicles.length; i++) {
        const eid = allVehicles[i];
        if (eid !== tankEid && select(eid)) {
            result.push(eid);
        }
    }
    
    if (result.length > limit) {
        shuffle(result);
        result.length = limit;
    }
    
    return result;
}


export const BULLET_DANGER_SPEED = 100;

export function findTankDangerBullets(tankEid: number, { physicalWorld } = GameDI) {
    const playerId = PlayerRef.id[tankEid];
    const position = {
        x: RigidBodyState.position.get(tankEid, 0),
        y: RigidBodyState.position.get(tankEid, 1),
    };
    const rotation = RigidBodyState.rotation[tankEid];
    const approximateColliderRadius = HeuristicsData.approxColliderRadius[tankEid];
    const tested = new Set<EntityId>();
    const result: EntityId[] = [];

    for (let j = 1; j < 5; j++) {
        const radius = 10 ** j;
        physicalWorld.intersectionsWithShape(
            position,
            rotation,
            new Ball(radius),
            (collider: Collider) => {
                const eid = getEntityIdByPhysicalId(collider.handle);

                if (tested.has(eid) || eid === 0 || PlayerRef.id[eid] === playerId) return true;

                tested.add(eid);

                const bulletPosition = RigidBodyState.position.getBatch(eid);
                const bulletVelocity = RigidBodyState.linvel.getBatch(eid);
                const dangerSpeed = hypot(bulletVelocity[0], bulletVelocity[1]) >= BULLET_DANGER_SPEED;
                if (!dangerSpeed) return true;

                const dangerTrajectory = hasIntersectionVectorAndCircle(
                    bulletPosition[0],
                    bulletPosition[1],
                    bulletVelocity[0],
                    bulletVelocity[1],
                    position.x,
                    position.y,
                    approximateColliderRadius * 2,
                );
                if (!dangerTrajectory) return true;

                const isDuplicate = hasSimilarTrajectory(
                    result,
                    bulletPosition,
                    bulletVelocity,
                );

                if (isDuplicate) return true;

                result.push(eid);

                return true;
            },
            undefined,
            createCollisionGroups(CollisionGroup.BULLET, CollisionGroup.BULLET),
        );
        if (result.length >= MAX_BULLETS) {
            break;
        }
    }

    if (result.length > MAX_BULLETS) {
        shuffle(result);
        result.length = MAX_BULLETS;
    }

    return result;
}

function hasSimilarTrajectory(
    bulletEids: number[],
    currentBulletPosition: Float64Array,
    currentBulletVelocity: Float64Array,
) {
    for (let i = 0; i < bulletEids.length; i++) {
        const bulletEid = bulletEids[i];
        const bulletPosition = RigidBodyState.position.getBatch(bulletEid);
        const bulletVelocity = RigidBodyState.linvel.getBatch(bulletEid);

        // Проверяем, что направление движения примерно совпадает
        // Нормализуем векторы скорости
        const currentBulletSpeed = hypot(
            currentBulletVelocity[0],
            currentBulletVelocity[1],
        );

        const bulletSpeed = hypot(
            bulletVelocity[0],
            bulletVelocity[1],
        );

        // Рассчитываем единичные векторы направления
        const currentBulletDir = [
            currentBulletVelocity[0] / currentBulletSpeed,
            currentBulletVelocity[1] / currentBulletSpeed,
        ];

        const bulletDir = [
            bulletVelocity[0] / bulletSpeed,
            bulletVelocity[1] / bulletSpeed,
        ];

        // Скалярное произведение единичных векторов
        const dotProduct = currentBulletDir[0] * bulletDir[0] + currentBulletDir[1] * bulletDir[1];

        // Если направления не совпадают, пули не могут быть на одной линии
        // Проверяем, что косинус угла между векторами близок к 1 или -1 (параллельны)
        if (Math.abs(dotProduct) < 0.98) { // Примерно в пределах ~11 градусов
            continue;
        }

        // Теперь проверяем, находится ли пуля на той же линии траектории
        // Вектор между позициями пуль
        const posDiff = [
            bulletPosition[0] - currentBulletPosition[0],
            bulletPosition[1] - currentBulletPosition[1],
        ];

        // Проекция вектора разности позиций на направление текущей пули
        const projection = posDiff[0] * currentBulletDir[0] + posDiff[1] * currentBulletDir[1];

        // Вектор проекции - это отрезок на линии траектории ближайший к позиции сравниваемой пули
        const projectedVector = [
            currentBulletDir[0] * projection,
            currentBulletDir[1] * projection,
        ];

        // Вычисляем вектор перпендикулярного расстояния (отклонение от линии)
        const perpVector = [
            posDiff[0] - projectedVector[0],
            posDiff[1] - projectedVector[1],
        ];

        // Длина этого вектора - это расстояние от пули до линии траектории
        const distance = hypot(
            perpVector[0],
            perpVector[1],
        );

        // Если расстояние меньше допустимого, значит пули на одной линии
        if (distance < 5) {
            return true;
        }
    }

    return false;
}

