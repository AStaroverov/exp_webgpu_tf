import { RigidBodyState } from '../../ECS/Components/Physical.ts';
import { BattleState, BULLET_DANGER_SPEED, getBattleState } from '../../ECS/Systems/RL/createTankInputTensorSystem.ts';
import { abs, acos, centerStep, hypot, max, min, PI, sin, smoothstep } from '../../../../../lib/math.ts';
import { TANK_RADIUS } from '../Common/consts.ts';
import { getMatrixTranslation, LocalTransform } from '../../../../../src/ECS/Components/Transform.ts';
import { Tank } from '../../ECS/Components/Tank.ts';
import { TankController } from '../../ECS/Components/TankController.ts';
import { ALLY_BUFFER, BULLET_BUFFER, ENEMY_BUFFER, TankInputTensor } from '../../ECS/Components/TankState.ts';
import { EntityId } from 'bitecs';
import { BULLET_SPEED } from '../../ECS/Components/Bullet.ts';
import { getTankHealth, getTankScore } from '../../ECS/Entities/Tank/TankUtils.ts';

const WEIGHTS = Object.freeze({
    WINNER: 5,
    DEATH: -5,

    TEAM: {
        SCORE: 0.2,
    },
    TEAM_MULTIPLIER: 1,

    COMMON: {
        HEALTH: 0.6,
    },
    COMMON_MULTIPLIER: 1,

    AIM: {
        QUALITY: 1.0,
        DISTANCE: 0.1,
        DISTANCE_PENALTY: -0.1,
        SHOOTING_ENEMIES: 1.0,
        SHOOTING_ALLIES_PENALTY: -1.0,
        SHOOTING_BAD_AIM: -0.2,
        SHOOTING_GOOD_AIM_PENALTY: -0.3,
    },
    AIM_MULTIPLIER: 1,

    MAP_BORDER: {
        BASE: 0.2,          // За нахождение в пределах карты
        PENALTY: -1.0,    // За выход за границы
    },
    MAP_BORDER_MULTIPLIER: 1,

    DISTANCE_KEEPING: {
        BASE: 0.2,          // За поддержание дистанции
        PENALTY: -0.4,      // За неудачную дистанцию
    },
    DISTANCE_KEEPING_MULTIPLIER: 1, // может быть несколько врагов

    BULLET_AVOIDANCE: {
        PENALTY: -0.6,
        AVOID_QUALITY: 0.6,
    },
    BULLET_AVOIDANCE_MULTIPLIER: 1,
});

// Структура для хранения многокомпонентных наград
export type ComponentRewards = {
    team: {
        score: number;          // Награда за разницу в здоровье
        total: number;         // Суммарная награда для команды
    }

    common: {
        health: number;
        total: number;
    };

    aim: {
        distance: number;        // Расстояние до цели
        accuracy: number;        // Точность прицеливания
        shootDecision: number;   // Решение о стрельбе
        total: number;           // Суммарная награда для головы прицеливания
    };

    // Награды для головы движения
    positioning: {
        enemiesPositioning: number;     // Позиционирование относительно врагов
        alliesPositioning: number;      // Позиционирование относительно союзников
        bulletAvoidance: number;       // Избегание опасности
        mapAwareness: number;    // Нахождение в пределах карты
        total: number;           // Суммарная награда для головы движения
    };

    // Общая суммарная награда
    totalReward: number;
}

function initializeRewards(): ComponentRewards {
    return {
        team: { score: 0, total: 0 },
        common: { health: 0, total: 0 },
        aim: { accuracy: 0, distance: 0, total: 0, shootDecision: 0 },
        positioning: {
            enemiesPositioning: 0,
            alliesPositioning: 0,
            bulletAvoidance: 0,
            mapAwareness: 0,
            total: 0,
        },
        totalReward: 0,
    };
}

const EPSILON = 1e-6; // Для избежания деления на ноль

export function calculateReward(
    tankEid: number,
    width: number,
    height: number,
    frame: number,
    isEnd: boolean,
    isWinner: boolean,
): number {
    const currentHealth = getTankHealth(tankEid);

    if (isEnd) {
        const score = getTankScore(tankEid) / frame;
        console.log('>> score', score);

        if (currentHealth <= 0) {
            return WEIGHTS.DEATH + score;
        }

        if (isWinner) {
            return WEIGHTS.WINNER + score;
        }

        return score;
    }

    const isShooting = TankController.shoot[tankEid] > 0;
    const [currentTankX, currentTankY] = RigidBodyState.position.getBatch(tankEid);
    // const [currentTankSpeedX, currentTankSpeedY] = RigidBodyState.linvel.getBatche(tankEid);
    const [currentTurretTargetX, currentTurretTargetY] = getMatrixTranslation(LocalTransform.matrix.getBatch(Tank.aimEid[tankEid]));
    // const currentShootings = TankController.shoot[tankEid] > 0;
    // const currentEnemies = findTankEnemiesEids(tankEid);
    // const currentAllies = findTankAlliesEids(tankEid);
    // const currentDangerBullets = findTankDangerBullets(tankEid);
    const currentBattleState = getBattleState(tankEid);

    const beforePredictEnemiesData = TankInputTensor.enemiesData.getBatch(tankEid);
    const beforePredictEnemiesEids = beforePredictEnemiesData.reduce((acc, v, i) => {
        if (i % ENEMY_BUFFER === 0 && v !== 0) acc.push(v);
        return acc;
    }, [] as EntityId[]);
    const beforePredictAlliesData = TankInputTensor.alliesData.getBatch(tankEid);
    const beforePredictAlliesEids = beforePredictAlliesData.reduce((acc, v, i) => {
        if (i % ALLY_BUFFER === 0 && v !== 0) acc.push(v);
        return acc;
    }, [] as EntityId[]);
    const beforePredictBulletsData = TankInputTensor.bulletsData.getBatch(tankEid);
    const beforePredictBulletsEids = beforePredictBulletsData.reduce((acc, v, i) => {
        if (i % BULLET_BUFFER === 0 && v !== 0) acc.push(v);
        return acc;
    }, [] as EntityId[]);

    const rewards = initializeRewards();

    rewards.team.score = getTeamAdvantageScore(currentBattleState);

    rewards.common.health = WEIGHTS.COMMON.HEALTH * currentHealth;

    rewards.positioning.mapAwareness = calculateTankMapAwarenessReward(
        width,
        height,
        currentTankX,
        currentTankY,
    );

    // 3. Анализ целей и вычисление награды за прицеливание
    const aimingResult = analyzeAiming(
        currentTankX,
        currentTankY,
        currentTurretTargetX,
        currentTurretTargetY,
        beforePredictEnemiesEids,
        beforePredictAlliesEids,
    );

    rewards.aim.accuracy = aimingResult.aimQualityReward;
    rewards.aim.distance = aimingResult.aimDistanceReward;
    rewards.aim.shootDecision = calculateShootingReward(
        isShooting,
        aimingResult.bestEnemyAimQuality,
        aimingResult.bestAlliesAimQuality,
    );

    const bulletAvoidanceResult = calculateBulletAvoidanceReward(
        currentTankX,
        currentTankY,
        beforePredictBulletsEids,
    );
    rewards.positioning.bulletAvoidance = bulletAvoidanceResult.reward;

    rewards.positioning.enemiesPositioning = calculateEnemyDistanceReward(
        currentTankX,
        currentTankY,
        beforePredictEnemiesEids,
    );

    rewards.positioning.alliesPositioning = calculateAllyDistanceReward(
        currentTankX,
        currentTankY,
        beforePredictAlliesEids,
    );

    // Рассчитываем итоговые значения
    rewards.team.total = WEIGHTS.TEAM_MULTIPLIER
        * (rewards.team.score);
    rewards.common.total = WEIGHTS.COMMON_MULTIPLIER
        * (rewards.common.health);
    rewards.aim.total = WEIGHTS.AIM_MULTIPLIER
        * (rewards.aim.accuracy + rewards.aim.distance + rewards.aim.shootDecision);
    rewards.positioning.total =
        (rewards.positioning.enemiesPositioning * WEIGHTS.DISTANCE_KEEPING_MULTIPLIER
            + rewards.positioning.alliesPositioning * WEIGHTS.DISTANCE_KEEPING_MULTIPLIER
            + rewards.positioning.bulletAvoidance * WEIGHTS.BULLET_AVOIDANCE_MULTIPLIER
            + rewards.positioning.mapAwareness * WEIGHTS.MAP_BORDER_MULTIPLIER);

    // Общая итоговая награда
    rewards.totalReward = rewards.team.total + rewards.common.total + rewards.aim.total + rewards.positioning.total;

    return rewards.totalReward;
}

export function getTeamAdvantageScore(
    state: BattleState,
    alpha = 1,
    beta = 1,
): number {
    const tanksCount = state.alliesCount + state.enemiesCount;
    const normCount = (state.alliesCount - state.enemiesCount) / tanksCount;
    const normHP = (state.alliesTotalHealth - state.enemiesTotalHealth) / tanksCount;

    // Combine the two normalised components
    return WEIGHTS.TEAM.SCORE * (alpha * normCount + beta * normHP);
}

function calculateTankMapAwarenessReward(
    width: number,
    height: number,
    x: number,
    y: number,
): number {
    const isInBounds = x >= 0 && x <= width && y >= 0 && y <= height;

    if (isInBounds) {
        return 0;
    } else {
        const dist = distanceToMap(width, height, x, y);
        return 0.8 * WEIGHTS.MAP_BORDER.PENALTY * smoothstep(0, 200, dist)
            + 0.2 * WEIGHTS.MAP_BORDER.PENALTY * smoothstep(0, 500, dist);
    }
}

function distanceToMap(width: number, height: number, x: number, y: number): number {
    if (x >= 0 && x <= width && y >= 0 && y <= height) {
        return 0;
    }

    const closestX = max(0, min(width, x));
    const closestY = max(0, min(height, y));
    const dx = x - closestX;
    const dy = y - closestY;

    return hypot(dx, dy);
}

function analyzeAiming(
    tankX: number,
    tankY: number,
    turretTargetX: number,
    turretTargetY: number,
    beforePredictEnemiesEids: number[],
    beforePredictAlliesEids: number[],
): {
    bestEnemyAimQuality: number;
    bestEnemyAimTargetId: number;
    bestAlliesAimQuality: number;
    aimQualityReward: number;
    aimDistanceReward: number;
} {
    let bestEnemyAimQuality = 0;
    let bestEnemyAimTargetId = 0;
    let bestEnemyDistance = 0;
    let bestAlliesAimQuality = 0;

    for (let i = 0; i < beforePredictEnemiesEids.length; i++) {
        const enemyId = beforePredictEnemiesEids[i];
        const enemyX = RigidBodyState.position.get(enemyId, 0);
        const enemyY = RigidBodyState.position.get(enemyId, 1);
        const enemyVX = RigidBodyState.linvel.get(enemyId, 0);
        const enemyVY = RigidBodyState.linvel.get(enemyId, 1);

        const distToEnemy = hypot(tankX - enemyX, tankY - enemyY);
        const timeDistToEnemy = distToEnemy / BULLET_SPEED;
        const futureEnemyX = enemyX + enemyVX * timeDistToEnemy;
        const futureEnemyY = enemyY + enemyVY * timeDistToEnemy;

        const aimQuality = computeAimQuality(tankX, tankY, turretTargetX, turretTargetY, futureEnemyX, futureEnemyY);

        // Отслеживаем лучшее прицеливание
        if (aimQuality > bestEnemyAimQuality) {
            bestEnemyAimQuality = aimQuality;
            bestEnemyAimTargetId = enemyId;
            bestEnemyDistance = hypot(tankX - enemyX, tankY - enemyY);
        }
    }

    // Анализируем всех видимых союзников для текущего состояния
    for (let i = 0; i < beforePredictAlliesEids.length; i++) {
        const allyId = beforePredictAlliesEids[i];
        const allyX = RigidBodyState.position.get(allyId, 0);
        const allyY = RigidBodyState.position.get(allyId, 1);
        const allyVX = RigidBodyState.linvel.get(allyId, 0);
        const allyVY = RigidBodyState.linvel.get(allyId, 1);

        const distToAlly = hypot(tankX - allyX, tankY - allyY);
        const timeDistToAlly = distToAlly / BULLET_SPEED;
        const futureAllyX = allyX + allyVX * timeDistToAlly;
        const futureAllyY = allyY + allyVY * timeDistToAlly;

        let aimQuality = computeAimQuality(tankX, tankY, turretTargetX, turretTargetY, futureAllyX, futureAllyY);

        if (distToAlly > bestEnemyDistance) {
            const distDiff = 1 - (distToAlly - bestEnemyDistance) / distToAlly;
            aimQuality *= distDiff;
        }

        bestAlliesAimQuality = max(bestAlliesAimQuality, aimQuality);
    }

    // Награда за качество прицеливания и дистанцию до цели
    const aimQualityReward = (bestEnemyAimQuality * WEIGHTS.AIM.QUALITY);

    // Награда за дистанцию прицеливания
    const turretTargetDistance = hypot(turretTargetX - tankX, turretTargetY - tankY);
    const aimDistanceReward =
        WEIGHTS.AIM.DISTANCE * (
            turretTargetDistance < 300
                ? smoothstep(TANK_RADIUS, 300, turretTargetDistance)
                : smoothstep(600, 300, turretTargetDistance)
        )
        + WEIGHTS.AIM.DISTANCE_PENALTY * smoothstep(TANK_RADIUS, 0, turretTargetDistance)
        + WEIGHTS.AIM.DISTANCE_PENALTY * smoothstep(600, 1000, turretTargetDistance);

    return {
        bestEnemyAimQuality,
        bestEnemyAimTargetId,
        aimQualityReward,
        aimDistanceReward,
        bestAlliesAimQuality,
    };
}

function computeAimQuality(
    tankX: number,
    tankY: number,
    turretTargetX: number,
    turretTargetY: number,
    enemyX: number,
    enemyY: number,
): number {
    // Вектор от танка к турели
    const tankToTurretTargetX = turretTargetX - tankX;
    const tankToTurretTargetY = turretTargetY - tankY;

    // Вектор от танка к противнику
    const tankToEnemyX = enemyX - tankX;
    const tankToEnemyY = enemyY - tankY;

    // Вычисляем длины векторов
    const tankToEnemyDist = hypot(tankToEnemyX, tankToEnemyY);
    const tankToTurretTargetDist = hypot(tankToTurretTargetX, tankToTurretTargetY);

    // Нормализованные векторы
    const turretNormX = tankToTurretTargetX / (tankToTurretTargetDist + EPSILON);
    const turretNormY = tankToTurretTargetY / (tankToTurretTargetDist + EPSILON);
    const enemyNormX = tankToEnemyX / (tankToEnemyDist + EPSILON);
    const enemyNormY = tankToEnemyY / (tankToEnemyDist + EPSILON);

    // Скалярное произведение нормализованных векторов
    const dotProduct = turretNormX * enemyNormX + turretNormY * enemyNormY;

    // Угол между векторами (в радианах)
    const angle = acos(max(-1, min(1, dotProduct)));

    // Вычисляем векторное произведение для определения знака (по часовой или против)
    const crossProduct = turretNormX * enemyNormY - turretNormY * enemyNormX;
    const signedAngle = crossProduct >= 0 ? angle : -angle;
    // Вычисляем качество прицеливания для прямого выстрела
    // Чем меньше угол, тем лучше прицеливание для прямого выстрела
    const angleAimQuality = smoothstep(PI / 4, 0, abs(signedAngle));

    // Вычисляем расстояние, на котором линия выстрела пройдет от противника (расстояние касательной)
    // Вычисляем перпендикулярное расстояние от линии выстрела до противника (это и есть расстояние касательной)
    const tangentialDistance = sin(angle) * tankToEnemyDist;
    // Вычисляем качество прицеливания для касательного выстрела
    // Награда за выстрел, проходящий на оптимальном расстоянии от противника
    const tangentialAimQuality = smoothstep(TANK_RADIUS * 1.5, 0, abs(tangentialDistance));

    return 0.2 * angleAimQuality + 0.8 * tangentialAimQuality;
}

function calculateShootingReward(
    isShooting: boolean,
    bestEnemyAimQuality: number,
    bestAlliesAimQuality: number,
): number {
    if (isShooting && bestAlliesAimQuality > bestEnemyAimQuality) {
        return WEIGHTS.AIM.SHOOTING_ALLIES_PENALTY;
    }

    const enoughAimQuality = 0.35;

    if (isShooting) {
        return bestEnemyAimQuality > enoughAimQuality
            ? WEIGHTS.AIM.SHOOTING_ENEMIES * bestEnemyAimQuality
            : WEIGHTS.AIM.SHOOTING_BAD_AIM * (1 - bestEnemyAimQuality);
    } else if (bestEnemyAimQuality > enoughAimQuality) {
        return WEIGHTS.AIM.SHOOTING_GOOD_AIM_PENALTY * smoothstep(0.8, 1.0, bestEnemyAimQuality);
    }

    return 0;
}

function calculateEnemyDistanceReward(
    tankX: number,
    tankY: number,
    beforePredictEnemiesEids: number[],
): number {
    let positioningReward = 0;

    // Анализируем всех видимых врагов для текущего состояния
    for (let i = 0; i < beforePredictEnemiesEids.length; i++) {
        const enemyId = beforePredictEnemiesEids[i];
        const enemyX = RigidBodyState.position.get(enemyId, 0);
        const enemyY = RigidBodyState.position.get(enemyId, 1);

        const distToEnemy = hypot(tankX - enemyX, tankY - enemyY);
        const minDist = TANK_RADIUS * 2;
        const maxDist = 600;

        if (distToEnemy < minDist) {
            // Штраф за слишком близкое расстояние
            const tooClosePenalty = smoothstep(minDist, 0, distToEnemy);
            positioningReward += tooClosePenalty * WEIGHTS.DISTANCE_KEEPING.PENALTY;
        } else if (distToEnemy <= maxDist) {
            // Награда за оптимальную дистанцию
            const optimalDistanceReward = centerStep(minDist, maxDist, distToEnemy);
            positioningReward += optimalDistanceReward * WEIGHTS.DISTANCE_KEEPING.BASE;
        } else {
            // Мягкий штраф за слишком большую дистанцию
            const tooFarPenalty = smoothstep(maxDist, maxDist * 1.5, distToEnemy) * WEIGHTS.DISTANCE_KEEPING.PENALTY;
            positioningReward += tooFarPenalty;
        }
    }

    return positioningReward / max(1, beforePredictEnemiesEids.length);
}

function calculateAllyDistanceReward(
    tankX: number,
    tankY: number,
    beforePredictAlliesEids: number[],
): number {
    let positioningReward = 0;

    for (let i = 0; i < beforePredictAlliesEids.length; i++) {
        const allyId = beforePredictAlliesEids[i];
        const allyX = RigidBodyState.position.get(allyId, 0);
        const allyY = RigidBodyState.position.get(allyId, 1);

        const distToAlly = hypot(tankX - allyX, tankY - allyY);
        const minDist = TANK_RADIUS * 2;

        if (distToAlly < minDist) {
            // Штраф за слишком близкое расстояние
            const tooClosePenalty = smoothstep(minDist, 0, distToAlly);
            positioningReward += tooClosePenalty * WEIGHTS.DISTANCE_KEEPING.PENALTY;
        }
    }

    return positioningReward / max(1, beforePredictAlliesEids.length);
}

function calculateBulletAvoidanceReward(
    tankX: number,
    tankY: number,
    beforePredictBulletsEids: number[],
): { reward: number; maxDangerLevel: number } {
    let reward = 0;
    let maxDangerLevel = 0;

    for (let i = 0; i < beforePredictBulletsEids.length; i++) {
        const bulletId = beforePredictBulletsEids[i];
        const bulletX = RigidBodyState.position.get(bulletId, 0);
        const bulletY = RigidBodyState.position.get(bulletId, 1);
        const bulletVx = RigidBodyState.linvel.get(bulletId, 0);
        const bulletVy = RigidBodyState.linvel.get(bulletId, 1);

        const bulletResult = analyzeBullet(
            tankX,
            tankY,
            bulletX,
            bulletY,
            bulletVx,
            bulletVy,
        );

        // Обновляем статистику
        reward += bulletResult.reward;
        maxDangerLevel = max(maxDangerLevel, bulletResult.dangerLevel);
    }

    return { reward, maxDangerLevel };
}

function analyzeBullet(
    tankX: number,
    tankY: number,
    bulletX: number,
    bulletY: number,
    bulletVx: number,
    bulletVy: number,
): { reward: number; dangerLevel: number } {
    // Анализируем текущую опасность пули
    const bulletSpeed = hypot(bulletVx, bulletVy);

    if (bulletSpeed < BULLET_DANGER_SPEED) return { reward: 0, dangerLevel: 0 };

    const bulletDirectionX = bulletVx / bulletSpeed;
    const bulletDirectionY = bulletVy / bulletSpeed;

    // Вектор от пули к танку
    const toTankX = tankX - bulletX;
    const toTankY = tankY - bulletY;

    // Определяем, движется ли пуля к танку
    const dotProduct = toTankX * bulletDirectionX + toTankY * bulletDirectionY;

    if (dotProduct < 0) return { reward: 0, dangerLevel: 0 };

    // Точка ближайшего прохождения пули к танку
    const closestPointX = bulletX + bulletDirectionX * dotProduct;
    const closestPointY = bulletY + bulletDirectionY * dotProduct;

    // Расстояние в точке наибольшего сближения
    const minDist = hypot(closestPointX - tankX, closestPointY - tankY);

    const dangerLevel = smoothstep(TANK_RADIUS * 1.5, TANK_RADIUS / 2, minDist);

    const reward = dangerLevel < 0.2
        ? (1 - dangerLevel) * WEIGHTS.BULLET_AVOIDANCE.AVOID_QUALITY
        : dangerLevel * WEIGHTS.BULLET_AVOIDANCE.PENALTY;

    return { reward, dangerLevel };
}