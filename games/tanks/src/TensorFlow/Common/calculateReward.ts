import {
    TANK_INPUT_TENSOR_BULLET_BUFFER,
    TANK_INPUT_TENSOR_ENEMY_BUFFER,
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';
import { RigidBodyState } from '../../ECS/Components/Physical.ts';
import {
    BULLET_DANGER_SPEED,
    findTankDangerBullets,
    findTankEnemies,
} from '../../ECS/Systems/RL/createTankInputTensorSystem.ts';
import { abs, centerStep, hypot, lerp, max, min, sign, smoothstep } from '../../../../../lib/math.ts';
import { TANK_RADIUS } from './consts.ts';
import { getMatrixTranslation, LocalTransform } from '../../../../../src/ECS/Components/Transform.ts';
import { getTankHealth, Tank } from '../../ECS/Components/Tank.ts';
import { TankController } from '../../ECS/Components/TankController.ts';
import { CONFIG } from '../PPO/Common/config.ts';
import { isVerboseLog } from './utils.ts';
import { clamp } from 'lodash-es';

// Константы для калибровки вознаграждений
let REWARD_WEIGHTS = {
    COMMON: {
        HEALTH: 0.3, // За выживание
        SURVIVAL: 1.0, // За выживание
    },
    COMMON_MULTIPLIER: 0.0,

    AIM: {
        QUALITY: 1.0,       // За точное прицеливание
        TRACKING: 0.2,      // За активное отслеживание врага
        DISTANCE: 0.1,      // За расстояние до цели
        MAP_AWARENESS: 0.02, // За нахождение в пределах карты
        NO_TARGET_PENALTY: -0.08, // За отсутствие целей
        TRACKING_PENALTY: -0.08, // За активное отслеживание врага
        DISTANCE_PENALTY: -0.08, // За расстояние до цели
        SHOOTING: 0.2,       // За стрельбу в цель
        SHOOTING_PENALTY: -0.08, // За стрельбу в пустоту
    },
    AIM_MULTIPLIER: 5.0,

    MAP_BORDER: {
        BASE: 1,          // За нахождение в пределах карты
        PENALTY: -1,      // За выход за границы
    },
    MAP_BORDER_MULTIPLIER: 3,

    DISTANCE_KEEPING: {
        BASE: 1.0,          // За поддержание дистанции
        PENALTY: -0.3,      // За слишком близкое приближение
    },
    DISTANCE_KEEPING_MULTIPLIER: 3,

    MOVEMENT: {
        BASE: 1,          // За базовое движение
        STRATEGIC: 1,     // За стратегическое движение
    },
    MOVEMENT_MULTIPLIER: 1.0,

    BULLET_AVOIDANCE: {
        PENALTY: -0.1,
        AVOID_QUALITY: 0.1,
    },
    BULLET_AVOIDANCE_MULTIPLIER: 0.0,
};

// Структура для хранения многокомпонентных наград
export interface ComponentRewards {
    common: {
        health: number;
        survival: number;
        total: number;
    };

    aim: {
        distance: number;        // Расстояние до цели
        accuracy: number;        // Точность прицеливания
        tracking: number;        // Активное отслеживание цели
        mapAwareness: number;    // Нахождение в пределах карты
        shootDecision: number;   // Решение о стрельбе
        total: number;           // Суммарная награда для головы прицеливания
    };

    // Награды для головы движения
    positioning: {
        movement: number;
        enemiesPositioning: number;     // Позиционирование относительно врагов
        bulletAvoidance: number;       // Избегание опасности
        mapAwareness: number;    // Нахождение в пределах карты
        total: number;           // Суммарная награда для головы движения
    };

    // Общая суммарная награда
    totalReward: number;
}

/**
 * Инициализация пустой структуры наград
 */
function initializeRewards(): ComponentRewards {
    return {
        common: { health: 0, survival: 0, total: 0 },
        aim: { accuracy: 0, tracking: 0, distance: 0, mapAwareness: 0, total: 0, shootDecision: 0 },
        positioning: { movement: 0, enemiesPositioning: 0, bulletAvoidance: 0, mapAwareness: 0, total: 0 },
        totalReward: 0,
    };
}

export function calculateReward(
    tankEid: number,
    width: number,
    height: number,
    step: number,
): ComponentRewards {
    // -- before predict --
    // const beforePredictHealth = TankInputTensor.health[tankEid];
    const [beforePredictTankX, beforePredictTankY] = TankInputTensor.position.getBatch(tankEid);
    // const [beforePredictTankSpeedX, beforePredictTankSpeedY] = TankInputTensor.speed.getBatche(tankEid);
    const [beforePredictTurretTargetX, beforePredictTurretTargetY] = TankInputTensor.turretTarget.getBatch(tankEid);
    const beforePredictEnemiesData = TankInputTensor.enemiesData.getBatch(tankEid);
    const beforePredictBulletsData = TankInputTensor.bulletsData.getBatch(tankEid);

    // -- current state --
    const moveDir = TankController.move[tankEid];
    const isShooting = TankController.shoot[tankEid] > 0;
    const currentHealth = getTankHealth(tankEid);
    const [currentTankX, currentTankY] = RigidBodyState.position.getBatch(tankEid);
    // const [currentTankSpeedX, currentTankSpeedY] = RigidBodyState.linvel.getBatche(tankEid);
    const [currentTurretTargetX, currentTurretTargetY] = getMatrixTranslation(LocalTransform.matrix.getBatch(Tank.aimEid[tankEid]));
    // const currentShootings = TankController.shoot[tankEid] > 0;
    const currentEnemies = Array.from(findTankEnemies(tankEid));
    const currentDangerBullets = Array.from(findTankDangerBullets(tankEid));

    // Инициализируем пустую структуру наград
    const rewards = initializeRewards();

    // 1. Награда за выживание
    rewards.common.health = currentHealth * REWARD_WEIGHTS.COMMON.HEALTH;
    rewards.common.survival = (step / CONFIG.episodeFrames) * REWARD_WEIGHTS.COMMON.SURVIVAL;

    // 2. Расчет награды за нахождение в пределах карты
    rewards.positioning.mapAwareness = calculateTankMapAwarenessReward(
        currentTankX,
        currentTankY,
        width,
        height,
    );

    // 3. Анализ целей и вычисление награды за прицеливание
    const aimingResult = analyzeAiming(
        currentTankX,
        currentTankY,
        currentTurretTargetX,
        currentTurretTargetY,
        currentEnemies,
        beforePredictTurretTargetX,
        beforePredictTurretTargetY,
        beforePredictEnemiesData,
    );

    rewards.aim.mapAwareness = calculateAimMapAwarenessReward(
        currentTurretTargetX,
        currentTurretTargetY,
        width,
        height,
    );
    rewards.aim.accuracy = aimingResult.aimQualityReward;
    rewards.aim.distance = aimingResult.aimDistanceReward;

    // 4. Награда за отслеживание целей и изменение прицела
    rewards.aim.tracking = calculateTrackingReward(
        aimingResult,
    );

    // 5. Награда за решение о стрельбе
    rewards.aim.shootDecision = calculateShootingReward(
        isShooting,
        aimingResult.bestAimQuality,
    );

    // 6. Награда за избегание пуль
    const bulletAvoidanceResult = calculateBulletAvoidanceReward(
        currentTankX,
        currentTankY,
        beforePredictTankX,
        beforePredictTankY,
        currentDangerBullets,
        beforePredictBulletsData,
    );
    rewards.positioning.bulletAvoidance = bulletAvoidanceResult.reward;

    // 7. Награда за движение и позиционирование
    const movementRewardResult = calculateMovementReward(
        moveDir,
        bulletAvoidanceResult.maxDangerLevel,
        aimingResult.hasTargets,
        aimingResult.closestEnemyDist,
    );

    rewards.positioning.movement = movementRewardResult.speed;
    rewards.positioning.enemiesPositioning = movementRewardResult.positioning;

    // Рассчитываем итоговые значения
    rewards.common.total = REWARD_WEIGHTS.COMMON_MULTIPLIER
        * (rewards.common.health + rewards.common.survival);
    rewards.aim.total = REWARD_WEIGHTS.AIM_MULTIPLIER
        * (rewards.aim.accuracy + rewards.aim.tracking + rewards.aim.distance + rewards.aim.mapAwareness + rewards.aim.shootDecision);
    rewards.positioning.total =
        (rewards.positioning.movement * REWARD_WEIGHTS.MOVEMENT_MULTIPLIER
            + rewards.positioning.enemiesPositioning * REWARD_WEIGHTS.DISTANCE_KEEPING_MULTIPLIER
            + rewards.positioning.bulletAvoidance * REWARD_WEIGHTS.BULLET_AVOIDANCE_MULTIPLIER
            + rewards.positioning.mapAwareness * REWARD_WEIGHTS.MAP_BORDER_MULTIPLIER);

    // Общая итоговая награда
    rewards.totalReward = rewards.common.total + rewards.aim.total + rewards.positioning.total;

    isVerboseLog() &&
    console.log(`[Reward] Tank ${ tankEid }
    aim: ${ rewards.aim.total }
    move: ${ rewards.positioning.total }
    total: ${ rewards.totalReward }
    `);

    return rewards;
}

/**
 * Расчет награды за нахождение в пределах карты
 */
function calculateTankMapAwarenessReward(
    x: number,
    y: number,
    width: number,
    height: number,
): number {
    if (x >= 0 && x <= width && y >= 0 && y <= height) {
        const borderDistance = min(
            x,
            y,
            width - x,
            height - y,
        );

        // Базовая награда за нахождение в пределах карты
        return REWARD_WEIGHTS.MAP_BORDER.BASE
            // Штраф за приближение к границе
            + REWARD_WEIGHTS.MAP_BORDER.PENALTY * (1 - smoothstep(0, 50, borderDistance));
    } else {
        // Вышел за границы карты
        return REWARD_WEIGHTS.MAP_BORDER.PENALTY;
    }
}

function calculateAimMapAwarenessReward(
    x: number,
    y: number,
    width: number,
    height: number,
): number {
    if (x >= 0 && x <= width && y >= 0 && y <= height) {
        return REWARD_WEIGHTS.AIM.MAP_AWARENESS;
    } else {
        return 0;
    }
}

/**
 * Анализ прицеливания и видимых врагов
 */
function analyzeAiming(
    tankX: number,
    tankY: number,
    turretTargetX: number,
    turretTargetY: number,
    currentEnemiesList: number[],
    prevTurretTargetX: number,
    prevTurretTargetY: number,
    beforePredictEnemiesData: Float64Array,
): {
    bestAimQuality: number;
    bestAimTargetId: number;
    aimQualityReward: number;
    aimDistanceReward: number;
    hasTargets: boolean;
    closestEnemyDist: number;
    prevBestAimQuality: number;
    prevBestAimTargetId: number;
} {
    let bestAimQuality = 0;
    let bestAimTargetId = 0;
    let prevBestAimQuality = 0;
    let prevBestAimTargetId = 0;
    let hasTargets = currentEnemiesList.length > 0;
    let closestEnemyDist = Number.MAX_VALUE;

    // Анализируем предыдущее прицеливание
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_ENEMIES; i++) {
        const enemyId = beforePredictEnemiesData[i * TANK_INPUT_TENSOR_ENEMY_BUFFER];

        if (enemyId === 0) continue;

        const prevEnemyX = beforePredictEnemiesData[i * TANK_INPUT_TENSOR_ENEMY_BUFFER + 1];
        const prevEnemyY = beforePredictEnemiesData[i * TANK_INPUT_TENSOR_ENEMY_BUFFER + 2];
        const prevAimQuality = computeAimQuality(prevTurretTargetX, prevTurretTargetY, prevEnemyX, prevEnemyY);

        if (prevAimQuality > prevBestAimQuality) {
            prevBestAimQuality = prevAimQuality;
            prevBestAimTargetId = enemyId;
        }
    }

    // Анализируем всех видимых врагов для текущего состояния
    for (let i = 0; i < currentEnemiesList.length; i++) {
        const enemyId = currentEnemiesList[i];
        const enemyX = RigidBodyState.position.get(enemyId, 0);
        const enemyY = RigidBodyState.position.get(enemyId, 1);

        const distToEnemy = hypot(tankX - enemyX, tankY - enemyY);

        // Обновляем дистанцию до ближайшего врага
        if (distToEnemy < closestEnemyDist) {
            closestEnemyDist = distToEnemy;
        }

        const currentAimQuality = computeAimQuality(turretTargetX, turretTargetY, enemyX, enemyY);

        // Отслеживаем лучшее прицеливание
        if (currentAimQuality > bestAimQuality) {
            bestAimQuality = currentAimQuality;
            bestAimTargetId = enemyId;
        }
    }

    // Награда за качество прицеливания и дистанцию до цели
    const aimQualityReward =
        (bestAimQuality * REWARD_WEIGHTS.AIM.QUALITY)
        + (bestAimTargetId === 0 ? REWARD_WEIGHTS.AIM.NO_TARGET_PENALTY : 0);

    // Награда за дистанцию прицеливания
    const turretTargetDistance = hypot(turretTargetX - tankX, turretTargetY - tankY);
    const aimDistanceReward =
        REWARD_WEIGHTS.AIM.DISTANCE * (
            turretTargetDistance < 300
                ? smoothstep(TANK_RADIUS, 300, turretTargetDistance)
                : smoothstep(300, 800, turretTargetDistance)
        )
        + REWARD_WEIGHTS.AIM.DISTANCE_PENALTY * smoothstep(TANK_RADIUS, 0, turretTargetDistance)
        + REWARD_WEIGHTS.AIM.DISTANCE_PENALTY * smoothstep(800, 1000, turretTargetDistance);

    return {
        bestAimQuality,
        bestAimTargetId,
        prevBestAimQuality,
        prevBestAimTargetId,
        aimQualityReward,
        aimDistanceReward,
        hasTargets,
        closestEnemyDist,
    };
}

function computeAimQuality(
    turretX: number,
    turretY: number,
    enemyX: number,
    enemyY: number,
): number {
    // Оценка точности прямого прицеливания
    const distFromTurretToEnemy = hypot(turretX - enemyX, turretY - enemyY);
    const directAimQuality =
        lerp(0, 0.02, smoothstep(1000, 0, distFromTurretToEnemy))
        + lerp(0, 0.02, smoothstep(TANK_RADIUS * 3, 0, distFromTurretToEnemy))
        + lerp(0, 0.96, smoothstep(TANK_RADIUS * 1.2, 0, distFromTurretToEnemy));

    return directAimQuality;
}

/**
 * Расчет награды за отслеживание целей
 */
function calculateTrackingReward(
    aimingResult: ReturnType<typeof analyzeAiming>,
): number {
    if (aimingResult.bestAimTargetId !== aimingResult.prevBestAimTargetId) {
        return REWARD_WEIGHTS.AIM.TRACKING_PENALTY;
    }

    if (aimingResult.bestAimQuality > 0.8) {
        return REWARD_WEIGHTS.AIM.TRACKING;
    }

    const { bestAimQuality, prevBestAimQuality } = aimingResult;
    // Вычисляем изменение комбинированного качества прицеливания
    const multipliedDelta = clamp((bestAimQuality - prevBestAimQuality) * 100, -1, 1);
    const deltaAimQuality = abs(multipliedDelta) < 0.001 ? 0 : multipliedDelta;

    // Награда за улучшение качества прицеливания
    const improvementReward = deltaAimQuality > 0
        ? deltaAimQuality * REWARD_WEIGHTS.AIM.TRACKING
        : 0;

    // Небольшой штраф за ухудшение прицеливания, но не такой сильный как награда за улучшение
    const deteriorationPenalty = deltaAimQuality < 0
        ? abs(deltaAimQuality) * REWARD_WEIGHTS.AIM.TRACKING_PENALTY
        : 0;

    const trackingReward = improvementReward + deteriorationPenalty;

    return trackingReward;
}

/**
 * Расчет награды за решение о стрельбе
 */
function calculateShootingReward(
    isShooting: boolean,
    bestAimQuality: number,
): number {
    let shootingReward = 0;

    if (isShooting) {
        // Плавная награда за стрельбу в зависимости от точности прицеливания
        shootingReward += bestAimQuality * REWARD_WEIGHTS.AIM.SHOOTING;
    } else if (bestAimQuality > 0.7) {
        // Небольшой штраф за отсутствие стрельбы при хорошем прицеливании
        shootingReward += REWARD_WEIGHTS.AIM.SHOOTING_PENALTY * smoothstep(0.8, 1.0, bestAimQuality);
    }

    return shootingReward;
}

/**
 * Расчет награды за движение
 */
function calculateMovementReward(
    moveDir: number,
    maxDangerLevel: number,
    hasTargets: boolean,
    closestEnemyDist: number,
): { speed: number; positioning: number } {
    let speedReward = 0;
    let positioningReward = 0;

    const moveFactor = abs(moveDir);

    // Базовая награда за движение
    speedReward += moveFactor * REWARD_WEIGHTS.MOVEMENT.BASE;

    // Стратегическое движение при наличии опасности
    if (maxDangerLevel > 0) {
        // Дополнительная награда за движение при наличии опасных пуль
        speedReward += moveFactor * REWARD_WEIGHTS.MOVEMENT.STRATEGIC;
    }

    // Награда за позиционирование относительно врагов
    if (hasTargets) {
        if (closestEnemyDist < TANK_RADIUS * 3) {
            // Штраф за слишком близкое расстояние
            const tooClosePenalty = 1 - smoothstep(0, TANK_RADIUS * 3, closestEnemyDist);
            positioningReward += tooClosePenalty * REWARD_WEIGHTS.DISTANCE_KEEPING.PENALTY;
        } else if (closestEnemyDist <= 600) {
            // Награда за оптимальную дистанцию
            const optimalDistanceReward = lerp(0.3, 1, centerStep(TANK_RADIUS * 3, 600, closestEnemyDist));
            positioningReward += optimalDistanceReward * REWARD_WEIGHTS.DISTANCE_KEEPING.BASE;
        } else {
            // Мягкий штраф за слишком большую дистанцию
            const tooFarPenalty = smoothstep(600, 800, closestEnemyDist) * REWARD_WEIGHTS.DISTANCE_KEEPING.PENALTY / 5;
            positioningReward += tooFarPenalty;
        }
    }

    return { speed: speedReward, positioning: positioningReward };
}

/**
 * Расчет награды за избегание пуль
 */
function calculateBulletAvoidanceReward(
    tankX: number,
    tankY: number,
    prevTankX: number,
    prevTankY: number,
    dangerBullets: number[],
    beforePredictBulletsData: Float64Array,
): { reward: number; maxDangerLevel: number } {
    let reward = 0;
    let maxDangerLevel = 0;

    // Создаем карту предыдущих пуль для быстрого поиска
    const prevBullets = new Map<number, { x: number, y: number, vx: number, vy: number }>();

    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_BULLETS; i++) {
        const id = beforePredictBulletsData[i * TANK_INPUT_TENSOR_BULLET_BUFFER];

        if (id === 0) continue;

        prevBullets.set(id, {
            x: beforePredictBulletsData[i * TANK_INPUT_TENSOR_BULLET_BUFFER + 1],
            y: beforePredictBulletsData[i * TANK_INPUT_TENSOR_BULLET_BUFFER + 2],
            vx: beforePredictBulletsData[i * TANK_INPUT_TENSOR_BULLET_BUFFER + 3],
            vy: beforePredictBulletsData[i * TANK_INPUT_TENSOR_BULLET_BUFFER + 4],
        });
    }

    // Анализируем каждую опасную пулю
    for (let i = 0; i < dangerBullets.length; i++) {
        const bulletId = dangerBullets[i];
        const bulletX = RigidBodyState.position.get(bulletId, 0);
        const bulletY = RigidBodyState.position.get(bulletId, 1);
        const bulletVx = RigidBodyState.linvel.get(bulletId, 0);
        const bulletVy = RigidBodyState.linvel.get(bulletId, 1);

        // Находим информацию о пуле в предыдущем состоянии
        const prevBulletInfo = prevBullets.get(bulletId);

        if (!prevBulletInfo) continue;

        // Анализируем пулю с учетом как текущего, так и предыдущего состояния
        const bulletResult = analyzeBullet(
            tankX,
            tankY,
            bulletX,
            bulletY,
            bulletVx,
            bulletVy,
            prevTankX,
            prevTankY,
            prevBulletInfo,
        );

        // Обновляем статистику
        reward += bulletResult.reward;
        maxDangerLevel = max(maxDangerLevel, bulletResult.dangerLevel);
    }

    return { reward, maxDangerLevel };
}

/**
 * Анализ опасности пули для танка
 */
function analyzeBullet(
    tankX: number,
    tankY: number,
    bulletX: number,
    bulletY: number,
    bulletVx: number,
    bulletVy: number,
    prevTankX: number,
    prevTankY: number,
    prevBulletInfo: { x: number, y: number, vx: number, vy: number },
): { reward: number; dangerLevel: number } {
    // Результаты анализа
    let reward = 0;
    let dangerLevel = 0;

    // Анализируем текущую опасность пули
    const bulletSpeed = hypot(bulletVx, bulletVy);
    if (bulletSpeed < BULLET_DANGER_SPEED) return { reward, dangerLevel };

    const bulletDirectionX = bulletVx / bulletSpeed;
    const bulletDirectionY = bulletVy / bulletSpeed;

    // Вектор от пули к танку
    const toTankX = tankX - bulletX;
    const toTankY = tankY - bulletY;

    // Определяем, движется ли пуля к танку
    const dotProduct = toTankX * bulletDirectionX + toTankY * bulletDirectionY;

    if (dotProduct < 0) return { reward, dangerLevel };

    // Точка ближайшего прохождения пули к танку
    const closestPointX = bulletX + bulletDirectionX * dotProduct;
    const closestPointY = bulletY + bulletDirectionY * dotProduct;

    // Расстояние в точке наибольшего сближения
    const minDist = hypot(closestPointX - tankX, closestPointY - tankY);

    if (minDist > TANK_RADIUS * 2) return { reward, dangerLevel };

    // Если есть информация о предыдущем состоянии пули, проверим было ли уклонение
    const prevBulletSpeed = hypot(prevBulletInfo.vx, prevBulletInfo.vy);

    const prevBulletDirX = prevBulletInfo.vx / prevBulletSpeed;
    const prevBulletDirY = prevBulletInfo.vy / prevBulletSpeed;

    // Вектор от предыдущей позиции пули к предыдущей позиции танка
    const prevToTankX = prevTankX - prevBulletInfo.x;
    const prevToTankY = prevTankY - prevBulletInfo.y;

    // Проверяем, двигалась ли пуля к танку в предыдущем состоянии
    const prevDotProduct = prevToTankX * prevBulletDirX + prevToTankY * prevBulletDirY;

    if (prevDotProduct > 0) {
        // Точка ближайшего прохождения пули к танку в предыдущем состоянии
        const prevClosestPointX = prevBulletInfo.x + prevBulletDirX * prevDotProduct;
        const prevClosestPointY = prevBulletInfo.y + prevBulletDirY * prevDotProduct;

        // Расстояние в точке наибольшего сближения в предыдущем состоянии
        const prevMinDist = hypot(prevClosestPointX - prevTankX, prevClosestPointY - prevTankY);

        // Награда или штраф за уклонение от пули
        const avoidanceQuality = sign(minDist - prevMinDist);
        reward += avoidanceQuality * REWARD_WEIGHTS.BULLET_AVOIDANCE.AVOID_QUALITY;
    }

    dangerLevel = smoothstep(TANK_RADIUS * 1.2, TANK_RADIUS / 2, minDist);

    return { reward, dangerLevel };
}