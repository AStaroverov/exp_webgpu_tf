import { EntityId } from 'bitecs';
import { clamp } from 'lodash';
import { abs, acos, cos, hypot, max, min, normalizeAngle, PI, sin, smoothstep } from '../../../../../lib/math.ts';
import { MAX_BULLET_SPEED } from '../../Game/ECS/Components/Bullet.ts';
import { HeuristicsData } from '../../Game/ECS/Components/HeuristicsData.ts';
import { RigidBodyState } from '../../Game/ECS/Components/Physical.ts';
import { Tank } from '../../Game/ECS/Components/Tank.ts';
import { TankController } from '../../Game/ECS/Components/TankController.ts';
import { getTankHealth, getTankScore } from '../../Game/ECS/Entities/Tank/TankUtils.ts';
import { ALLY_BUFFER, ENEMY_BUFFER, TankInputTensor } from '../../Pilots/Components/TankState.ts';
import { BattleState, getBattleState } from '../../Pilots/Utils/snapshotTankInputTensor.ts';

export const GAME_OVER_REWARD_MULTIPLIER = 5;
export const getFramePenalty = (frame: number) => -clamp(Math.log10(1 + frame / 15), 0, 3) / 30;

const WEIGHTS = ({
    STATE_MULTIPLIER: 1,
    ACTION_MULTIPLIER: 2,

    // ACTION REWARD
    COMMON: {
        SCORE: 2,
        HEALTH: 1,
    },
    COMMON_MULTIPLIER: 1,

    TEAM: {
        SCORE: 1,
    },
    TEAM_MULTIPLIER: 1,

    // STATE REWARD
    AIM: {
        QUALITY: 2,
        BAD_QUALITY_PENALTY: -0.25,
        BAD_SHOOTING_PENALTY: -1,
        ALLIES_SHOOTING_PENALTY: -1,
    },
    AIM_MULTIPLIER: 1,

    MAP_BORDER: {
        PENALTY: -1,
    },
    MAP_BORDER_MULTIPLIER: 3,

    DISTANCE: {
        MIN_PENALTY: -1,
    },
    DISTANCE_KEEPING_MULTIPLIER: 1,

    MOVING: {
        PENALTY_SPEED: -1,
    },
    MOVING_MULTIPLIER: 1,
});

function initializeStateRewards() {
    return {
        aim: { quality: 0, shootDecision: 0, total: 0 },
        moving: { speed: 0, total: 0 },
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
    tankEid: number,
    width: number,
    height: number,
    strictness: number,
): number {
    const isShooting = TankController.shoot[tankEid] > 0;
    const moveDir = TankController.move[tankEid];
    const rotationDir = TankController.rotation[tankEid];
    const [currentTankX, currentTankY] = RigidBodyState.position.getBatch(tankEid);
    // const [currentTankSpeedX, currentTankSpeedY] = RigidBodyState.linvel.getBatche(tankEid);
    const turretRotation = RigidBodyState.rotation[Tank.turretEId[tankEid]];
    // const currentShootings = TankController.shoot[tankEid] > 0;
    // const currentEnemies = findTankEnemiesEids(tankEid);
    // const currentAllies = findTankAlliesEids(tankEid);
    // const currentDangerBullets = findTankDangerBullets(tankEid);

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
    // const beforePredictBulletsData = TankInputTensor.bulletsData.getBatch(tankEid);
    // const beforePredictBulletsEids = beforePredictBulletsData.reduce((acc, v, i) => {
    //     if (i % BULLET_BUFFER === 0 && v !== 0) acc.push(v);
    //     return acc;
    // }, [] as EntityId[]);

    const rewards = initializeStateRewards();

    rewards.moving.speed = calculateMovingReward(moveDir, rotationDir);

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
        turretRotation,
        beforePredictEnemiesEids,
        beforePredictAlliesEids,
    );

    rewards.aim.quality = aimingResult.bestEnemyAimQuality > 0
        ? aimingResult.bestEnemyAimQuality * WEIGHTS.AIM.QUALITY
        : WEIGHTS.AIM.BAD_QUALITY_PENALTY;

    rewards.aim.shootDecision = strictness
        * calculateShootingPenalty(
            isShooting,
            aimingResult.bestEnemyAimQuality,
            aimingResult.bestAlliesAimQuality,
        );

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
        + rewards.aim.total
        + rewards.moving.total
        + rewards.positioning.total);

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
        team: { score: 0, total: 0 },
        common: { score: 0, health: 0, total: 0 },
    };
}

export function calculateActionReward(tankEid: number): number {
    const currentScore = getTankScore(tankEid);
    const currentHealth = getTankHealth(tankEid);
    const currentBattleState = getBattleState(tankEid);

    const rewards = initializeActionRewards();

    rewards.team.score = getTeamAdvantageScore(currentBattleState);
    rewards.common.score = WEIGHTS.COMMON.SCORE * currentScore / 10;
    rewards.common.health = WEIGHTS.COMMON.HEALTH * currentHealth;

    rewards.team.total = WEIGHTS.TEAM_MULTIPLIER
        * (rewards.team.score);
    rewards.common.total = WEIGHTS.COMMON_MULTIPLIER
        * (rewards.common.health + rewards.common.score);

    const totalReward = rewards.team.total + rewards.common.total;

    if (!Number.isFinite(totalReward)) {
        console.error(`
            team: ${rewards.team.total.toFixed(2)},
            common: ${rewards.common.total.toFixed(2)},
        `);
        throw new Error('Rewards are not finite:');
    }

    return WEIGHTS.ACTION_MULTIPLIER * totalReward;
}

function calculateMovingReward(moveDir: number, rotationDir: number): number {
    const absSumDir = min(abs(moveDir) + abs(rotationDir) / 2, 1);
    const minLimit = 0.2;

    return absSumDir > minLimit
        ? 0
        : WEIGHTS.MOVING.PENALTY_SPEED * (minLimit - absSumDir) / minLimit;
}

export function getTeamAdvantageScore(state: BattleState): number {
    const normCount = (state.alliesCount - state.enemiesCount) / 2;
    const normHP = (state.alliesTotalHealth - state.enemiesTotalHealth);

    return WEIGHTS.TEAM.SCORE * (normCount + normHP);
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

        const distToEnemy = hypot(tankX - enemyX, tankY - enemyY);
        const timeDistToEnemy = distToEnemy / MAX_BULLET_SPEED;
        const futureEnemyX = enemyX + enemyVX * timeDistToEnemy;
        const futureEnemyY = enemyY + enemyVY * timeDistToEnemy;
        const enemyColliderRadius = HeuristicsData.approxColliderRadius[enemyId];

        const aimQuality = computeAimQuality(tankX, tankY, turretRotation, futureEnemyX, futureEnemyY, enemyColliderRadius);

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
        const timeDistToAlly = distToAlly / MAX_BULLET_SPEED;
        const futureAllyX = allyX + allyVX * timeDistToAlly;
        const futureAllyY = allyY + allyVY * timeDistToAlly;
        const allyColliderRadius = HeuristicsData.approxColliderRadius[allyId];

        let aimQuality = computeAimQuality(tankX, tankY, turretRotation, futureAllyX, futureAllyY, allyColliderRadius);

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
    tankX: number,
    tankY: number,
    turretRotation: number,
    enemyX: number,
    enemyY: number,
    colliderRadius: number,
): number {
    // 1. Вектор «танк → враг»
    const tankToEnemyX = enemyX - tankX;
    const tankToEnemyY = enemyY - tankY;
    const tankToEnemyDist = hypot(tankToEnemyX, tankToEnemyY);

    // Нормализуем
    const enemyNormX = tankToEnemyX / (tankToEnemyDist + EPSILON);
    const enemyNormY = tankToEnemyY / (tankToEnemyDist + EPSILON);

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
    const tangentialDistance = sin(angle) * tankToEnemyDist;  // расстояние от луча до центра врага
    const tangentialAimQuality = smoothstep(colliderRadius * 1.5, 0, abs(tangentialDistance));

    // 5. Итог
    return 0.2 * angleAimQuality + 0.8 * tangentialAimQuality;
}

function calculateShootingPenalty(
    isShooting: boolean,
    bestEnemyAimQuality: number,
    bestAlliesAimQuality: number,
): number {
    if (isShooting && bestAlliesAimQuality > bestEnemyAimQuality) {
        return WEIGHTS.AIM.ALLIES_SHOOTING_PENALTY;
    }

    if (isShooting && bestEnemyAimQuality < 0.35) {
        return WEIGHTS.AIM.BAD_SHOOTING_PENALTY * (1 - bestEnemyAimQuality);
    }

    return 0;
}

function calculateEnemyDistanceReward(
    tankX: number,
    tankY: number,
    beforePredictEnemiesEids: number[],
): number {
    let positioningReward = 0;

    for (let i = 0; i < beforePredictEnemiesEids.length; i++) {
        const enemyId = beforePredictEnemiesEids[i];
        const enemyX = RigidBodyState.position.get(enemyId, 0);
        const enemyY = RigidBodyState.position.get(enemyId, 1);
        const colliderRadius = HeuristicsData.approxColliderRadius[enemyId];

        const distToEnemy = hypot(tankX - enemyX, tankY - enemyY);
        const minDist = colliderRadius * 2;

        if (distToEnemy < minDist) {
            const tooClosePenalty = smoothstep(minDist, 0, distToEnemy);
            positioningReward += tooClosePenalty * WEIGHTS.DISTANCE.MIN_PENALTY;
        }
    }

    return positioningReward;
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
        const colliderRadius = HeuristicsData.approxColliderRadius[allyId];

        const distToAlly = hypot(tankX - allyX, tankY - allyY);
        const minDist = colliderRadius * 2;

        if (distToAlly < minDist) {
            const tooClosePenalty = smoothstep(minDist, 0, distToAlly);
            positioningReward += tooClosePenalty * WEIGHTS.DISTANCE.MIN_PENALTY;
        }
    }

    return positioningReward;
}
