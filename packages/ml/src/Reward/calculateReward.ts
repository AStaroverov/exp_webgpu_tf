import { EntityId } from 'bitecs';
import { clamp } from 'lodash';
import { abs, acos, cos, hypot, max, min, normalizeAngle, PI, sin, smoothstep } from '../../../../lib/math.ts';
import { MAX_BULLET_SPEED } from '../../../tanks/src/Game/ECS/Components/Bullet.ts';
import { HeuristicsData } from '../../../tanks/src/Game/ECS/Components/HeuristicsData.ts';
import { RigidBodyState } from '../../../tanks/src/Game/ECS/Components/Physical.ts';
import { Tank } from '../../../tanks/src/Game/ECS/Components/Tank.ts';
import { getTankHealth, getTankScore } from '../../../tanks/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { ALLY_BUFFER, ENEMY_BUFFER, TankInputTensor } from '../../../tanks/src/Pilots/Components/TankState.ts';
import { LEARNING_STEPS } from '../../../ml-common/consts.ts';
import { TurretController } from '../../../tanks/src/Game/ECS/Components/TurretController.ts';

export const getFramePenalty = (frame: number) =>
    -clamp(Math.log10(1 + frame), 0, 3) / 100;

export const getFinalReward = (successRatio: number, networkVersion: number) => 
    successRatio * (0.5 * (0.3 + 0.7 * clamp(networkVersion / LEARNING_STEPS, 0, 1)));

const WEIGHTS = ({
    STATE_MULTIPLIER: 1, // it's often reward and work as learning base principle
    ACTION_MULTIPLIER: 2, // it's rare reward and work as path to win

    // ACTION REWARD
    COMMON: {
        SCORE: 1,
        HEALTH: 1,
    },
    COMMON_MULTIPLIER: 1,

    // STATE REWARD
    AIM: {
        QUALITY: 1,
        BAD_QUALITY_PENALTY: -0.25,

        SHOOTING_REWARD: 1,
        NO_SHOOTING_PENALTY: -0.5,
        MISS_SHOOTING_PENALTY: -0.5,
        ALLIES_SHOOTING_PENALTY: -0.5,
    },
    AIM_MULTIPLIER: 1,

    MAP_BORDER: {
        PENALTY: -1,
    },
    MAP_BORDER_MULTIPLIER: 5,

    DISTANCE: {
        MIN_PENALTY: -0.5,
    },
    DISTANCE_KEEPING_MULTIPLIER: 1,

    MOVING: {
        SPEED: 1,
    },
    MOVING_MULTIPLIER: 1,
});

function initializeStateRewards() {
    return {
        aim: {quality: 0, shootDecision: 0, total: 0},
        moving: {speed: 0, total: 0},
        positioning: {
            enemiesPositioning: 0,
            alliesPositioning: 0,
            mapAwareness: 0,
            total: 0,
        },
    };
}

const EPSILON = 1e-6;

export function calculateStateReward(
    vehicleEid: number,
    width: number,
    height: number,
    strictness: number
): number {
    const isShooting = TurretController.shoot[Tank.turretEId[vehicleEid]] > 0;
    // const moveDir = VehicleController.move[vehicleEid];
    // const rotationDir = VehicleController.rotation[vehicleEid];
    const [currentVehicleX, currentVehicleY] = RigidBodyState.position.getBatch(vehicleEid);
    const [currentVehicleSpeedX, currentVehicleSpeedY] = RigidBodyState.linvel.getBatch(vehicleEid);
    const turretRotation = RigidBodyState.rotation[Tank.turretEId[vehicleEid]];
    // const currentShootings = VehicleController.shoot[vehicleEid] > 0;
    // const currentEnemies = findTankEnemiesEids(vehicleEid);
    // const currentAllies = findTankAlliesEids(vehicleEid);
    // const currentDangerBullets = findTankDangerBullets(vehicleEid);

    const beforePredictEnemiesData = TankInputTensor.enemiesData.getBatch(vehicleEid);
    const beforePredictEnemiesEids = beforePredictEnemiesData.reduce((acc, v, i) => {
        if (i % ENEMY_BUFFER === 0 && v !== 0) acc.push(v);
        return acc;
    }, [] as EntityId[]);
    const beforePredictAlliesData = TankInputTensor.alliesData.getBatch(vehicleEid);
    const beforePredictAlliesEids = beforePredictAlliesData.reduce((acc, v, i) => {
        if (i % ALLY_BUFFER === 0 && v !== 0) acc.push(v);
        return acc;
    }, [] as EntityId[]);
    // const beforePredictBulletsData = TankInputTensor.bulletsData.getBatch(vehicleEid);
    // const beforePredictBulletsEids = beforePredictBulletsData.reduce((acc, v, i) => {
    //     if (i % BULLET_BUFFER === 0 && v !== 0) acc.push(v);
    //     return acc;
    // }, [] as EntityId[]);

    const rewards = initializeStateRewards();

    rewards.moving.speed = calculateMovingReward(currentVehicleSpeedX, currentVehicleSpeedY);

    rewards.positioning.mapAwareness = calculateVehicleMapAwarenessReward(
        width,
        height,
        currentVehicleX,
        currentVehicleY,
        HeuristicsData.approxColliderRadius[vehicleEid]
    );

    // 3. Анализ целей и вычисление награды за прицеливание
    const aimingResult = analyzeAiming(
        currentVehicleX,
        currentVehicleY,
        turretRotation,
        beforePredictEnemiesEids,
        beforePredictAlliesEids,
    );

    rewards.aim.quality = aimingResult.bestEnemyAimQuality > 0
        ? aimingResult.bestEnemyAimQuality * WEIGHTS.AIM.QUALITY
        : WEIGHTS.AIM.BAD_QUALITY_PENALTY;

    rewards.aim.shootDecision = calculateShootingReward(
        isShooting,
        aimingResult.bestEnemyAimQuality,
        aimingResult.bestAlliesAimQuality,
        max(strictness, 0.3),
    );

    rewards.positioning.enemiesPositioning = calculateEnemyDistanceReward(
        currentVehicleX,
        currentVehicleY,
        beforePredictEnemiesEids,
    );

    rewards.positioning.alliesPositioning = calculateAllyDistanceReward(
        currentVehicleX,
        currentVehicleY,
        beforePredictAlliesEids,
    );

    rewards.aim.total = WEIGHTS.AIM_MULTIPLIER
        * (rewards.aim.quality
            + rewards.aim.shootDecision);
    rewards.moving.total = WEIGHTS.MOVING_MULTIPLIER
        * (rewards.moving.speed);
    rewards.positioning.total =
        (rewards.positioning.mapAwareness * WEIGHTS.MAP_BORDER_MULTIPLIER
            + rewards.positioning.alliesPositioning * WEIGHTS.DISTANCE_KEEPING_MULTIPLIER
            + rewards.positioning.enemiesPositioning * WEIGHTS.DISTANCE_KEEPING_MULTIPLIER);

    const totalReward = (
        rewards.aim.total
        + rewards.moving.total
        + rewards.positioning.total
    );

    // console.log('>>', `
    //     team: ${ rewards.team.total.toFixed(2) },
    //     common: ${ rewards.common.total.toFixed(2) },
    //     aim: ${ rewards.aim.total.toFixed(2) },
    //     moving: ${ rewards.moving.total.toFixed(2) },
    //     positioning: ${ rewards.positioning.total.toFixed(2) },
    // `);

    if (!Number.isFinite(totalReward)) {
        console.error(`
            aim: ${rewards.aim.total.toFixed(2)},
            moving: ${rewards.moving.total.toFixed(2)},
            positioning: ${rewards.positioning.total.toFixed(2)},
        `);
        throw new Error('Rewards are not finite:');
    }

    return WEIGHTS.STATE_MULTIPLIER * totalReward;
}


function initializeActionRewards() {
    return {
        common: {score: 0, health: 0, total: 0},
    };
}

export function calculateActionReward(vehicleEid: number): number {
    const currentScore = getTankScore(vehicleEid);
    const currentHealth = getTankHealth(vehicleEid);

    const rewards = initializeActionRewards();

    rewards.common.score = WEIGHTS.COMMON.SCORE * currentScore * 0.33;
    rewards.common.health = WEIGHTS.COMMON.HEALTH * currentHealth * 10;

    rewards.common.total = WEIGHTS.COMMON_MULTIPLIER
        * (rewards.common.health + rewards.common.score);

    const totalReward = rewards.common.total;

    if (!Number.isFinite(totalReward)) {
        console.error(`
            common: ${rewards.common.total.toFixed(2)},
        `);
        throw new Error('Rewards are not finite:');
    }

    return WEIGHTS.ACTION_MULTIPLIER * totalReward;
}

function calculateMovingReward(linvelX: number, linvelY: number): number {
    const speed = Math.hypot(linvelX, linvelY);
    return WEIGHTS.MOVING.SPEED * speed / 100
}

function calculateVehicleMapAwarenessReward(
    width: number,
    height: number,
    x: number,
    y: number,
    minGap: number,
): number {
    const gapW = max(width * 0.1, minGap);
    const gapH = max(height * 0.1, minGap);
    const isInBounds = x >= gapW && x <= (width - gapW) && y >= gapH && y <= (height - gapH);

    if (isInBounds) {
        return 0;
    } else {
        const dist = distanceToMap(gapW, gapH, width - gapW, height - gapH, x, y);
        const score = WEIGHTS.MAP_BORDER.PENALTY * smoothstep(0, 100, dist);
        return score;
    }
}

function distanceToMap(
    minX: number, minY: number,
    maxX: number, maxY: number,
    x: number, y: number
): number {
    const closestX = max(minX, min(maxX, x));
    const closestY = max(minY, min(maxY, y));
    const dx = x - closestX;
    const dy = y - closestY;

    return hypot(dx, dy);
}

function analyzeAiming(
    vehicleX: number,
    vehicleY: number,
    turretRotation: number,
    beforePredictEnemiesEids: number[],
    beforePredictAlliesEids: number[],
): {
    bestEnemyAimQuality: number;
    bestEnemyAimTargetId: number;
    bestAlliesAimQuality: number;
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

        const distToEnemy = hypot(vehicleX - enemyX, vehicleY - enemyY);
        const timeDistToEnemy = distToEnemy / MAX_BULLET_SPEED;
        const futureEnemyX = enemyX + enemyVX * timeDistToEnemy;
        const futureEnemyY = enemyY + enemyVY * timeDistToEnemy;
        const enemyColliderRadius = HeuristicsData.approxColliderRadius[enemyId];

        const aimQuality = computeAimQuality(vehicleX, vehicleY, turretRotation, futureEnemyX, futureEnemyY, enemyColliderRadius);

        // Отслеживаем лучшее прицеливание
        if (aimQuality > bestEnemyAimQuality) {
            bestEnemyAimQuality = aimQuality;
            bestEnemyAimTargetId = enemyId;
            bestEnemyDistance = hypot(vehicleX - enemyX, vehicleY - enemyY);
        }
    }

    // Анализируем всех видимых союзников для текущего состояния
    for (let i = 0; i < beforePredictAlliesEids.length; i++) {
        const allyId = beforePredictAlliesEids[i];
        const allyX = RigidBodyState.position.get(allyId, 0);
        const allyY = RigidBodyState.position.get(allyId, 1);
        const allyVX = RigidBodyState.linvel.get(allyId, 0);
        const allyVY = RigidBodyState.linvel.get(allyId, 1);

        const distToAlly = hypot(vehicleX - allyX, vehicleY - allyY);
        const timeDistToAlly = distToAlly / MAX_BULLET_SPEED;
        const futureAllyX = allyX + allyVX * timeDistToAlly;
        const futureAllyY = allyY + allyVY * timeDistToAlly;
        const allyColliderRadius = HeuristicsData.approxColliderRadius[allyId];

        let aimQuality = computeAimQuality(vehicleX, vehicleY, turretRotation, futureAllyX, futureAllyY, allyColliderRadius);

        if (bestEnemyAimQuality > 0 && distToAlly > bestEnemyDistance) {
            const distDiff = 1 - (distToAlly - bestEnemyDistance) / distToAlly;
            aimQuality *= distDiff;
        }

        bestAlliesAimQuality = max(bestAlliesAimQuality, aimQuality);
    }

    return {
        bestEnemyAimQuality,
        bestEnemyAimTargetId,
        bestAlliesAimQuality,
    };
}

function computeAimQuality(
    vehicleX: number,
    vehicleY: number,
    turretRotation: number,
    enemyX: number,
    enemyY: number,
    colliderRadius: number,
): number {
    // 1. Вектор «vehicle → враг»
    const vehicleToEnemyX = enemyX - vehicleX;
    const vehicleToEnemyY = enemyY - vehicleY;
    const vehicleToEnemyDist = hypot(vehicleToEnemyX, vehicleToEnemyY);

    // Нормализуем
    const enemyNormX = vehicleToEnemyX / (vehicleToEnemyDist + EPSILON);
    const enemyNormY = vehicleToEnemyY / (vehicleToEnemyDist + EPSILON);

    // 2. Вектор направления ствола
    //    turretDir = (cos φ, sin φ), φ = tankRot + turretRot
    const turretAngle = normalizeAngle(turretRotation - Math.PI / 2);
    const turretNormX = cos(turretAngle);
    const turretNormY = sin(turretAngle);

    // 3. Сравниваем два норм-вектора
    const dotProduct = turretNormX * enemyNormX + turretNormY * enemyNormY;
    if (dotProduct < 0) return 0;                 // враг позади ствола

    const angle = acos(max(-1, min(1, dotProduct)));          // |угол| ∈ [0, π]

    // знак угла нужен только если вы где-то его используете дальше
    const crossProduct = turretNormX * enemyNormY - turretNormY * enemyNormX;
    const signedAngle = crossProduct >= 0 ? angle : -angle;

    // Чем меньше угол, тем лучше прямое прицеливание
    const angleAimQuality = smoothstep(PI / 4, 0, abs(signedAngle));

    // 4. Оценка «касательного» выстрела
    const tangentialDistance = sin(angle) * vehicleToEnemyDist;  // расстояние от луча до центра врага
    const tangentialAimQuality = smoothstep(colliderRadius * 1.5, 0, abs(tangentialDistance));

    // 5. Итог
    return 0.2 * angleAimQuality + 0.8 * tangentialAimQuality;
}

function calculateShootingReward(
    isShooting: boolean,
    bestEnemyAimQuality: number,
    bestAlliesAimQuality: number,
    strictness: number
): number {
    if (isShooting && bestAlliesAimQuality > bestEnemyAimQuality && bestAlliesAimQuality > 0.5) {
        return strictness * WEIGHTS.AIM.ALLIES_SHOOTING_PENALTY * (bestAlliesAimQuality - 0.5) * (1 / (1 - 0.5));
    }

    if (!isShooting && bestEnemyAimQuality > 0.7) {
        return strictness * WEIGHTS.AIM.NO_SHOOTING_PENALTY * (bestEnemyAimQuality - 0.7) * (1 / (1 - 0.7));
    }

    if (isShooting && bestEnemyAimQuality > 0.5) {
        return WEIGHTS.AIM.SHOOTING_REWARD * bestEnemyAimQuality * (1 / (1 - 0.5));
    }

    return 0;
}

function calculateEnemyDistanceReward(
    vehicleX: number,
    vehicleY: number,
    beforePredictEnemiesEids: number[],
): number {
    let positioningReward = 0;

    for (let i = 0; i < beforePredictEnemiesEids.length; i++) {
        const enemyId = beforePredictEnemiesEids[i];
        const enemyX = RigidBodyState.position.get(enemyId, 0);
        const enemyY = RigidBodyState.position.get(enemyId, 1);
        const colliderRadius = HeuristicsData.approxColliderRadius[enemyId];

        const distToEnemy = hypot(vehicleX - enemyX, vehicleY - enemyY);
        const minDist = colliderRadius * 2;

        if (distToEnemy < minDist) {
            const tooClosePenalty = smoothstep(minDist, 0, distToEnemy);
            positioningReward += tooClosePenalty * WEIGHTS.DISTANCE.MIN_PENALTY;
        }
    }

    return positioningReward;
}

function calculateAllyDistanceReward(
    vehicleX: number,
    vehicleY: number,
    beforePredictAlliesEids: number[],
): number {
    let positioningReward = 0;

    for (let i = 0; i < beforePredictAlliesEids.length; i++) {
        const allyId = beforePredictAlliesEids[i];
        const allyX = RigidBodyState.position.get(allyId, 0);
        const allyY = RigidBodyState.position.get(allyId, 1);
        const colliderRadius = HeuristicsData.approxColliderRadius[allyId];

        const distToAlly = hypot(vehicleX - allyX, vehicleY - allyY);
        const minDist = colliderRadius * 2;

        if (distToAlly < minDist) {
            const tooClosePenalty = smoothstep(minDist, 0, distToAlly);
            positioningReward += tooClosePenalty * WEIGHTS.DISTANCE.MIN_PENALTY;
        }
    }

    return positioningReward;
}
