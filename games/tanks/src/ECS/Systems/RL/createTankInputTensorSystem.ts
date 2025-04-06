import { GameDI } from '../../../DI/GameDI.ts';
import { getTankHealth, Tank, TANK_APPROXIMATE_COLLISION_RADIUS } from '../../Components/Tank.ts';
import { MAX_ALLIES, MAX_BULLETS, MAX_ENEMIES, TankInputTensor } from '../../Components/TankState.ts';
import { getEntityIdByPhysicalId, RigidBodyState } from '../../Components/Physical.ts';
import { hypot } from '../../../../../../lib/math.ts';
import { Ball, Collider } from '@dimforge/rapier2d-simd/rapier';
import { CollisionGroup, createCollisionGroups } from '../../../Physical/createRigid.ts';
import { EntityId, query } from 'bitecs';
import { Player } from '../../Components/Player.ts';
import { getMatrixTranslation, LocalTransform } from '../../../../../../src/ECS/Components/Transform.ts';
import { hasIntersectionVectorAndCircle } from '../../../Utils/intersections.ts';
import { shuffle } from '../../../../../../lib/shuffle.ts';
import { TenserFlowDI } from '../../../DI/TenserFlowDI.ts';
import { Team } from '../../Components/Team.ts';

export function createTankInputTensorSystem({ world } = GameDI) {
    return () => {
        if (!TenserFlowDI.enabled || !TenserFlowDI.shouldCollectState) return;

        const tankEids = query(world, [Tank, TankInputTensor, RigidBodyState]);

        TankInputTensor.resetEnemiesCoords();
        TankInputTensor.resetAlliesCoords();
        TankInputTensor.resetBulletsCoords();

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];
            const tankTeamId = Team.id[tankEid];

            // Set battlefield data
            const allEnemiesEids = tankEids.filter((eid) => Team.id[eid] !== tankTeamId);
            const allEnemiesHealth = allEnemiesEids.reduce((acc, eid) => acc + getTankHealth(eid), 0);
            const allAlliesEids = tankEids.filter((eid) => Team.id[eid] === tankTeamId);
            const allAlliesHealth = allAlliesEids.reduce((acc, eid) => acc + getTankHealth(eid), 0);

            TankInputTensor.setBattlefieldData(
                tankEid,
                allEnemiesEids.length,
                allEnemiesHealth,
                allAlliesEids.length,
                allAlliesHealth,
            );

            // Set tank data
            const health = getTankHealth(tankEid);
            const linvel = RigidBodyState.linvel.getBatch(tankEid);
            const position = RigidBodyState.position.getBatch(tankEid);
            const aimLocal = LocalTransform.matrix.getBatch(Tank.aimEid[tankEid]);

            TankInputTensor.setTankData(
                tankEid,
                health,
                linvel,
                position,
                getMatrixTranslation(aimLocal),
            );

            // Find closest enemies
            const enemiesEids = findTankEnemies(tankEid);

            for (let j = 0; j < enemiesEids.length; j++) {
                const enemyEid = enemiesEids[j];

                TankInputTensor.setEnemiesData(
                    tankEid,
                    j,
                    enemyEid,
                    getTankHealth(enemyEid),
                    RigidBodyState.position.getBatch(enemyEid),
                    RigidBodyState.linvel.getBatch(enemyEid),
                    getMatrixTranslation(LocalTransform.matrix.getBatch(Tank.aimEid[enemyEid])),
                );
            }

            // Find closest allies
            const alliesEids = findTankAllies(tankEid);

            for (let j = 0; j < alliesEids.length; j++) {
                const allyEid = alliesEids[j];

                TankInputTensor.setAlliesData(
                    tankEid,
                    j,
                    allyEid,
                    getTankHealth(allyEid),
                    RigidBodyState.position.getBatch(allyEid),
                    RigidBodyState.linvel.getBatch(allyEid),
                    getMatrixTranslation(LocalTransform.matrix.getBatch(Tank.aimEid[allyEid])),
                );
            }

            // Find closest bullets
            const bulletsEids = findTankDangerBullets(tankEid);

            for (let j = 0; j < bulletsEids.length; j++) {
                const bulletEid = bulletsEids[j];

                TankInputTensor.setBulletsData(
                    tankEid,
                    j,
                    bulletEid,
                    RigidBodyState.position.getBatch(bulletEid),
                    RigidBodyState.linvel.getBatch(bulletEid),
                );
            }
        }
    };
}

export function findTankEnemies(tankEid: EntityId) {
    const tankTeamId = Team.id[tankEid];
    return findTankNeighbours(tankEid, MAX_ENEMIES, (eid: EntityId) => tankTeamId !== Team.id[eid]);
}

export function findTankAllies(tankEid: EntityId) {
    const tankTeamId = Team.id[tankEid];
    return findTankNeighbours(tankEid, MAX_ALLIES, (eid: EntityId) => tankTeamId === Team.id[eid]);
}

export function findTankNeighbours(tankEid: EntityId, limit: number, select: (eid: EntityId) => boolean, { physicalWorld } = GameDI) {
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
            createCollisionGroups(CollisionGroup.TANK_BASE, CollisionGroup.TANK_BASE),
        );
        if (result.length >= limit) {
            break;
        }
    }

    return result;
}


export const BULLET_DANGER_SPEED = 100;

export function findTankDangerBullets(tankEid: number, { physicalWorld } = GameDI) {
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

                if (tested.has(eid) || eid === 0 || Player.id[eid] === tankEid) return true;

                tested.add(eid);

                const bulletPosition = RigidBodyState.position.getBatch(eid);
                const bulletVelocity = RigidBodyState.linvel.getBatch(eid);
                const dangerSpeed = hypot(bulletPosition[0], bulletPosition[1]) >= BULLET_DANGER_SPEED;
                if (!dangerSpeed) return true;

                const dangerTrajectory = hasIntersectionVectorAndCircle(
                    bulletPosition[0],
                    bulletPosition[1],
                    bulletVelocity[0],
                    bulletVelocity[1],
                    position.x,
                    position.y,
                    TANK_APPROXIMATE_COLLISION_RADIUS * 2,
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

