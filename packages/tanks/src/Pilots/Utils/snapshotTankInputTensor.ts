import { Ball, Collider, Ray, Vector2 } from '@dimforge/rapier2d-simd';
import { EntityId, hasComponent, query } from 'bitecs';
import { cos, hypot, PI, sin } from '../../../../../lib/math.ts';
import { shuffle } from '../../../../../lib/shuffle.ts';
import { GameDI } from '../../Game/DI/GameDI.ts';
import { getEntityIdByPhysicalId, RigidBodyRef, RigidBodyState } from '../../Game/ECS/Components/Physical.ts';
import { PlayerRef } from '../../Game/ECS/Components/PlayerRef.ts';
import { Tank } from '../../Game/ECS/Components/Tank.ts';
import { Vehicle } from '../../Game/ECS/Components/Vehicle.ts';
import { TeamRef } from '../../Game/ECS/Components/TeamRef.ts';
import { GameMap } from '../../Game/ECS/Entities/GameMap.ts';
import { CollisionGroup, createCollisionGroups } from '../../Game/Physical/createRigid.ts';
import { hasIntersectionVectorAndCircle } from '../../Game/Utils/intersections.ts';
import {
    MAX_ALLIES,
    MAX_BULLETS,
    MAX_ENEMIES,
    TankInputTensor,
    ENV_RAYS_FORWARD,
    ENV_RAYS_BACKWARD,
    ENV_RAYS_LEFT,
    ENV_RAYS_RIGHT,
    ENV_RAY_LENGTH,
    ENV_RAY_SECTOR_ANGLE,
    RayHitType,
    TURRET_RAY_LENGTH,
} from '../Components/TankState.ts';

import { HeuristicsData } from '../../Game/ECS/Components/HeuristicsData.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';
import { Parent } from '../../Game/ECS/Components/Parent.ts';
import { Shape, ShapeKind } from 'renderer/src/ECS/Components/Shape.ts';
import { ALL_VEHICLE_PARTS_MASK, CollisionGroupConfig } from '../../Game/Config/physics.ts';

// Temp arrays for offset-adjusted positions
const tempPosition = new Float64Array(2);
const tempEnemyPosition = new Float64Array(2);
const tempAllyPosition = new Float64Array(2);
const tempBulletPosition = new Float64Array(2);

// Reusable ray objects
const rayOrigin = new Vector2(0, 0);
const rayDir = new Vector2(0, 0);

// Collision mask for environment rays: obstacles + vehicle parts
const ENV_RAY_COLLISION_MASK = CollisionGroupConfig.OBSTACLE | ALL_VEHICLE_PARTS_MASK;

export function snapshotTankInputTensor({ world } = GameDI) {
    const vehicleEids = query(world, [Vehicle, TankInputTensor, RigidBodyState]);
    const offsetX = GameMap.offsetX;
    const offsetY = GameMap.offsetY;

    TankInputTensor.resetEnemiesCoords();
    TankInputTensor.resetAlliesCoords();
    TankInputTensor.resetBulletsCoords();
    TankInputTensor.resetEnvRaysData();
    TankInputTensor.resetTurretRaysData();

    for (let i = 0; i < vehicleEids.length; i++) {
        const vehicleEid = vehicleEids[i];

        // Set vehicle data (with offset-adjusted position)
        const health = getTankHealth(vehicleEid);
        const position = RigidBodyState.position.getBatch(vehicleEid);
        tempPosition[0] = position[0] - offsetX;
        tempPosition[1] = position[1] - offsetY;
        const rotation = RigidBodyState.rotation[vehicleEid];
        const linvel = RigidBodyState.linvel.getBatch(vehicleEid);
        const approximateColliderRadius = HeuristicsData.approxColliderRadius[vehicleEid];
        const turretPosition = RigidBodyState.position.getBatch(Tank.turretEId[vehicleEid]);
        const turretRotation = RigidBodyState.rotation[Tank.turretEId[vehicleEid]];

        TankInputTensor.setTankData(
            vehicleEid,
            health,
            tempPosition,
            rotation,
            linvel,
            turretRotation,
            approximateColliderRadius,
        );

        const myTeamId = TeamRef.id[vehicleEid];

        // Find closest enemies
        const enemiesEids = findTankEnemiesEids(vehicleEid);

        for (let j = 0; j < enemiesEids.length; j++) {
            const enemyEid = enemiesEids[j];
            const enemyPosition = RigidBodyState.position.getBatch(enemyEid);
            tempEnemyPosition[0] = enemyPosition[0] - offsetX;
            tempEnemyPosition[1] = enemyPosition[1] - offsetY;

            TankInputTensor.setEnemiesData(
                vehicleEid,
                j,
                enemyEid,
                getTankHealth(enemyEid),
                tempEnemyPosition,
            );
        }

        // Find closest allies
        const alliesEids = findTankAlliesEids(vehicleEid);

        for (let j = 0; j < alliesEids.length; j++) {
            const allyEid = alliesEids[j];
            const allyPosition = RigidBodyState.position.getBatch(allyEid);
            tempAllyPosition[0] = allyPosition[0] - offsetX;
            tempAllyPosition[1] = allyPosition[1] - offsetY;

            TankInputTensor.setAlliesData(
                vehicleEid,
                j,
                allyEid,
                getTankHealth(allyEid),
                tempAllyPosition,
            );
        }

        // Find closest bullets
        const bulletsEids = findTankDangerBullets(vehicleEid);

        for (let j = 0; j < bulletsEids.length; j++) {
            const bulletEid = bulletsEids[j];
            const bulletPosition = RigidBodyState.position.getBatch(bulletEid);
            tempBulletPosition[0] = bulletPosition[0] - offsetX;
            tempBulletPosition[1] = bulletPosition[1] - offsetY;

            TankInputTensor.setBulletsData(
                vehicleEid,
                j,
                bulletEid,
                tempBulletPosition,
                RigidBodyState.linvel.getBatch(bulletEid),
            );
        }

        castEnvironmentRays(
            vehicleEid,
            myTeamId,
            position[0],
            position[1],
            rotation - PI / 2,
        );

        castTurretRays(
            vehicleEid,
            myTeamId,
            turretPosition[0],
            turretPosition[1],
            turretRotation - PI / 2,
        );
    }
}

/**
 * Creates a filter predicate that excludes colliders belonging to the specified vehicle
 */
function createExcludeOwnVehiclePredicate(
    vehicleEid: EntityId,
    { world } = GameDI,
): (collider: Collider) => boolean {
    return (collider: Collider): boolean => {
        const rigidBody = collider.parent();
        if (!rigidBody) return true;
        
        const eid = getEntityIdByPhysicalId(rigidBody.handle);
        if (eid === 0) return true;
        
        // Check if this entity is the vehicle itself
        if (eid === vehicleEid) return false;
        
        // Check if this entity belongs to the vehicle (via Parent chain)
        const ownerVehicleEid = findVehicleFromPart(eid, world);
        if (ownerVehicleEid === vehicleEid) return false;
        
        return true;
    };
}

/**
 * Cast 12 rays around the agent: 5 forward, 5 backward, 1 left, 1 right
 * Forward/backward groups cover sector angle, left/right are single perpendicular rays
 */
function castEnvironmentRays(
    vehicleEid: EntityId,
    myTeamId: number,
    posX: number,
    posY: number,
    rotation: number,
    { physicalWorld } = GameDI,
) {
    rayOrigin.x = posX;
    rayOrigin.y = posY;

    const filterPredicate = createExcludeOwnVehiclePredicate(vehicleEid);

    // Ray sectors: [centerAngle, rayCount, indexOffset]
    const sectors: [number, number, number][] = [
        // Forward
        [rotation, ENV_RAYS_FORWARD, 0],                                                            
        // Backward
        [rotation + Math.PI, ENV_RAYS_BACKWARD, ENV_RAYS_FORWARD],                                  
        // Left
        [rotation - Math.PI / 2, ENV_RAYS_LEFT, ENV_RAYS_FORWARD + ENV_RAYS_BACKWARD],              
        // Right
        [rotation + Math.PI / 2, ENV_RAYS_RIGHT, ENV_RAYS_FORWARD + ENV_RAYS_BACKWARD + ENV_RAYS_LEFT],
    ];

    for (const [centerAngle, rayCount, indexOffset] of sectors) {
        // For single rays, just cast at center angle; for multiple rays, spread across sector
        const startAngle = rayCount === 1 ? centerAngle : centerAngle - ENV_RAY_SECTOR_ANGLE / 2;
        const angleStep = rayCount === 1 ? 0 : ENV_RAY_SECTOR_ANGLE / (rayCount - 1);

        for (let i = 0; i < rayCount; i++) {
            const angle = startAngle + i * angleStep;
            rayDir.x = cos(angle);
            rayDir.y = sin(angle);
            const ray = new Ray(rayOrigin, rayDir);
            const hit = physicalWorld.castRay(
                ray,
                ENV_RAY_LENGTH,
                true,
                undefined,
                createCollisionGroups(CollisionGroup.ALL, ENV_RAY_COLLISION_MASK),
                undefined,
                undefined,
                filterPredicate,
            );

            if (hit == null) continue;
            
            const eid = getEntityIdByPhysicalId(hit.collider.handle);
            const rayHit = processRayHit(eid, myTeamId);
            TankInputTensor.setEnvRayData(
                vehicleEid,
                indexOffset + i,
                rayHit.hitType,
                rayHit.x,
                rayHit.y,
                rayHit.radius,
                hit.timeOfImpact,
            );
        }
    }
}

/**
 * Cast ray in turret direction
 */
function castTurretRays(
    vehicleEid: EntityId,
    myTeamId: number,
    posX: number,
    posY: number,
    turretRotation: number,
    { physicalWorld } = GameDI,
) {
    rayOrigin.x = posX;
    rayOrigin.y = posY;
    rayDir.x = cos(turretRotation);
    rayDir.y = sin(turretRotation);
    // Create filter to exclude own vehicle parts
    const filterPredicate = createExcludeOwnVehiclePredicate(vehicleEid);
    const ray = new Ray(rayOrigin, rayDir);
    const hit = physicalWorld.castRay(
        ray,
        TURRET_RAY_LENGTH,
        true,
        undefined, // filterFlags
        createCollisionGroups(CollisionGroup.ALL, ENV_RAY_COLLISION_MASK),
        undefined, // filterExcludeCollider
        undefined, // filterExcludeRigidBody
        filterPredicate,
    );

    if (hit) {
        const collider = hit.collider;
        const eid = getEntityIdByPhysicalId(collider.handle);
        
        // Determine hit type
        const hitType = getHitType(eid, myTeamId);
        
        // Calculate aiming error - angle from turret center to target center
        const aimingErrorDegrees = calculateAimingError(
            posX,
            posY, 
            turretRotation, 
            eid, 
            hitType,
        );
        
        TankInputTensor.setTurretRayData(vehicleEid, 0, hitType, hit.timeOfImpact, aimingErrorDegrees);
    }
}

/**
 * Calculate aiming error in degrees - angle between turret direction and direction to target center
 * Positive = target is to the right, Negative = target is to the left
 */
function calculateAimingError(
    posX: number,
    posY: number,
    turretRotation: number,
    targetEid: EntityId,
    hitType: RayHitType,
    { world } = GameDI,
): number {
    if (hitType === RayHitType.NONE) return 0;
    
    // Get target center position
    let targetX: number;
    let targetY: number;
    
    if (hitType === RayHitType.ENEMY_VEHICLE || hitType === RayHitType.ALLY_VEHICLE) {
        // For vehicle parts, get the vehicle center
        const vehicleEid = findVehicleFromPart(targetEid, world);
        if (vehicleEid === 0) return 0;
        [targetX, targetY] = RigidBodyState.position.getBatch(vehicleEid);
    } else {
        // For obstacles, use the obstacle position
        [targetX, targetY] = RigidBodyState.position.getBatch(targetEid);
    }
    
    // Calculate angle from our position to target center
    const dx = targetX - posX;
    const dy = targetY - posY;
    const angleToTarget = Math.atan2(dy, dx);
    
    // Calculate angle difference (normalize to -PI to PI)
    let angleDiff = angleToTarget - turretRotation;
    while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
    while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;
    
    // Convert to degrees
    const RAD_TO_DEG = 180 / Math.PI;
    return angleDiff * RAD_TO_DEG;
}

// Reusable result object to avoid allocations
const rayHitResult = { hitType: RayHitType.NONE as RayHitType, x: 0, y: 0, radius: 0 };

/**
 * Process ray hit and extract entity data (returns reusable object - do not store reference!)
 * @param eid - Entity ID of the hit collider
 * @param myTeamId - Team ID of the vehicle casting the ray (to determine enemy vs ally)
 */
function processRayHit(eid: EntityId, myTeamId: number, { world } = GameDI): typeof rayHitResult {
    const collisionGroup = getCollisionGroupFromEntity(eid);
    rayHitResult.hitType = RayHitType.NONE;
    rayHitResult.x = 0;
    rayHitResult.y = 0;
    rayHitResult.radius = 0;

    if (collisionGroup === 0) {
        return rayHitResult;
    }

    if (collisionGroup & ALL_VEHICLE_PARTS_MASK) {
        // It's a vehicle part - traverse parent to get vehicle
        const vehicleEid = findVehicleFromPart(eid, world);
        if (vehicleEid === 0) return rayHitResult;

        const pos = RigidBodyState.position.getBatch(vehicleEid);
        const hitTeamId = TeamRef.id[vehicleEid];
        rayHitResult.hitType = hitTeamId === myTeamId ? RayHitType.ALLY_VEHICLE : RayHitType.ENEMY_VEHICLE;
        rayHitResult.x = pos[0];
        rayHitResult.y = pos[1];
        rayHitResult.radius = HeuristicsData.approxColliderRadius[vehicleEid];
        return rayHitResult;
    }
    
    if (collisionGroup & CollisionGroupConfig.OBSTACLE) {
        // It's an obstacle - get its position and approximate radius
        const pos = RigidBodyState.position.getBatch(eid);
        rayHitResult.hitType = RayHitType.OBSTACLE;
        rayHitResult.x = pos[0];
        rayHitResult.y = pos[1];
        rayHitResult.radius = getApproxRadius(eid);
        return rayHitResult;
    }

    return rayHitResult;
}

/**
 * Get hit type from entity
 * @param eid - Entity ID of the hit collider
 * @param myTeamId - Team ID of the vehicle casting the ray (to determine enemy vs ally)
 */
function getHitType(eid: EntityId, myTeamId: number, { world } = GameDI): RayHitType {
    if (eid === 0) return RayHitType.NONE;

    const collisionGroup = getCollisionGroupFromEntity(eid);
    
    if (collisionGroup & ALL_VEHICLE_PARTS_MASK) {
        const vehicleEid = findVehicleFromPart(eid, world);
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
function findVehicleFromPart(partEid: EntityId, world: any): EntityId {
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

/**
 * Get approximate collider radius from entity shape
 */
function getApproxRadius(eid: EntityId): number {
    const kind = Shape.kind[eid];
    
    if (kind === ShapeKind.Circle) {
        return Shape.values.get(eid, 0); // radius
    } else if (kind === ShapeKind.Rectangle) {
        const width = Shape.values.get(eid, 0);
        const height = Shape.values.get(eid, 1);
        return Math.max(width, height) / 2;
    }
    
    // Fallback: use HeuristicsData if available
    return HeuristicsData.approxColliderRadius[eid] || 10;
}

export function findTankEnemiesEids(tankEid: EntityId) {
    const tankTeamId = TeamRef.id[tankEid];
    return findTankNeighboursEids(tankEid, MAX_ENEMIES, (eid: EntityId) => tankTeamId !== TeamRef.id[eid]);
}

export function findTankAlliesEids(tankEid: EntityId) {
    const tankTeamId = TeamRef.id[tankEid];
    return findTankNeighboursEids(tankEid, MAX_ALLIES, (eid: EntityId) => tankTeamId === TeamRef.id[eid]);
}

export function findTankNeighboursEids(tankEid: EntityId, limit: number, select: (eid: EntityId) => boolean, { physicalWorld } = GameDI) {
    // Find enemies
    const position = {
        x: RigidBodyState.position.get(tankEid, 0),
        y: RigidBodyState.position.get(tankEid, 1),
    };
    const rotation = RigidBodyState.rotation[tankEid];
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
                if (tested.has(eid) || eid === 0 || tankEid === eid) return true;

                tested.add(eid);

                if (!select(eid)) return true;

                result.push(eid);

                return result.length < limit;
            },
            undefined,
            createCollisionGroups(CollisionGroup.VEHICALE_BASE, CollisionGroup.VEHICALE_BASE),
        );
        if (result.length >= limit) {
            break;
        }
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

