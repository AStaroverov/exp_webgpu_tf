import { Ball, Collider } from '@dimforge/rapier2d-simd';
import { EntityId, query } from 'bitecs';
import { hypot } from '../../../../../lib/math.ts';
import { shuffle } from '../../../../../lib/shuffle.ts';
import { GameDI } from '../../Game/DI/GameDI.ts';
import { getEntityIdByPhysicalId, RigidBodyState } from '../../Game/ECS/Components/Physical.ts';
import { PlayerRef } from '../../Game/ECS/Components/PlayerRef.ts';
import { Tank } from '../../Game/ECS/Components/Tank.ts';
import { Vehicle } from '../../Game/ECS/Components/Vehicle.ts';
import { TeamRef } from '../../Game/ECS/Components/TeamRef.ts';
import { GameMap } from '../../Game/ECS/Entities/GameMap.ts';
import { CollisionGroup, createCollisionGroups } from '../../Game/Physical/createRigid.ts';
import { hasIntersectionVectorAndCircle } from '../../Game/Utils/intersections.ts';
import { MAX_ALLIES, MAX_BULLETS, MAX_ENEMIES, TankInputTensor } from '../Components/TankState.ts';

import { HeuristicsData } from '../../Game/ECS/Components/HeuristicsData.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';

// Temp arrays for offset-adjusted positions
const tempPosition = new Float64Array(2);
const tempEnemyPosition = new Float64Array(2);
const tempAllyPosition = new Float64Array(2);
const tempBulletPosition = new Float64Array(2);

export function snapshotTankInputTensor({ world } = GameDI) {
    const vehicleEids = query(world, [Vehicle, TankInputTensor, RigidBodyState]);
    const offsetX = GameMap.offsetX;
    const offsetY = GameMap.offsetY;

    TankInputTensor.resetEnemiesCoords();
    TankInputTensor.resetAlliesCoords();
    TankInputTensor.resetBulletsCoords();

    for (let i = 0; i < vehicleEids.length; i++) {
        const vehicleEid = vehicleEids[i];

        // Set vehicle data (with offset-adjusted position)
        const health = getTankHealth(vehicleEid);
        const position = RigidBodyState.position.getBatch(vehicleEid);
        tempPosition[0] = position[0] - offsetX;
        tempPosition[1] = position[1] - offsetY;
        const rotation = RigidBodyState.rotation[vehicleEid];
        const linvel = RigidBodyState.linvel.getBatch(vehicleEid);
        const turretRotation = RigidBodyState.rotation[Tank.turretEId[vehicleEid]];
        const approximateColliderRadius = HeuristicsData.approxColliderRadius[vehicleEid];

        TankInputTensor.setTankData(
            vehicleEid,
            health,
            tempPosition,
            rotation,
            linvel,
            turretRotation,
            approximateColliderRadius,
        );

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
                RigidBodyState.rotation[enemyEid],
                RigidBodyState.linvel.getBatch(enemyEid),
                RigidBodyState.rotation[Tank.turretEId[enemyEid]],
                HeuristicsData.approxColliderRadius[enemyEid],
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
                RigidBodyState.rotation[allyEid],
                RigidBodyState.linvel.getBatch(allyEid),
                RigidBodyState.rotation[Tank.turretEId[allyEid]],
                HeuristicsData.approxColliderRadius[allyEid],
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
    }
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

