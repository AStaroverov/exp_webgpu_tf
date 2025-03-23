import {
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
import { TANK_MAX_SPEED, TANK_RADIUS } from './consts.ts';
import { getMatrixTranslation, LocalTransform } from '../../../../../src/ECS/Components/Transform.ts';
import { getTankHealth, Tank } from '../../ECS/Components/Tank.ts';
import { TankController } from '../../ECS/Components/TankController.ts';
import { isVerboseLog } from './utils.ts';
import { CONFIG } from '../PPO/Common/config.ts';

// Константы для калибровки вознаграждений
let REWARD_WEIGHTS = {
    COMMON: {
        HEALTH: 0.3, // За выживание
        SURVIVAL: 1.0, // За выживание
    },

    AIM: {
        QUALITY: 1.0,       // За точное прицеливание
        TRACKING: 1.0,      // За активное отслеживание врага
        DISTANCE: 1.0,      // За расстояние до цели
        MAP_AWARENESS: 0.1, // За нахождение в пределах карты
        NO_TARGET_PENALTY: -0.1, // За отсутствие целей
        TRACKING_PENALTY: -0.2, // За активное отслеживание врага
        DISTANCE_PENALTY: -0.2, // За расстояние до цели
    },

    MAP_BORDER: {
        BASE: 0.2,          // За нахождение в пределах карты
        PENALTY: -0.2,      // За выход за границы
    },

    SHOOTING: {
        AIMED: 1.0,         // За прицельную стрельбу
        AIMED_PENALTY: -0.1, // Штраф за стрельбу в пустоту
    },

    DISTANCE_KEEPING: {
        BASE: 1.0,          // За поддержание дистанции
        PENALTY: -0.2,      // За слишком близкое приближение
    },

    MOVEMENT: {
        BASE: 0.2,          // За базовое движение
        STRATEGIC: 0.5,     // За стратегическое движение
    },

    BULLET_AVOIDANCE: {
        PENALTY: -0.1,
        AVOID_QUALITY: 0.1,
    },
};

// Структура для хранения многокомпонентных наград
export interface ComponentRewards {
    common: {
        health: number;
        survival: number;
        total: number;
    };
    // Награды для головы стрельбы
    shoot: {
        shootDecision: number;   // Решение о стрельбе
        total: number;           // Суммарная награда для головы стрельбы
    };

    // Награды для головы движения
    movement: {
        speed: number;           // Скорость движения
        enemiesPositioning: number;     // Позиционирование относительно врагов
        bulletAvoidance: number;       // Избегание опасности
        mapAwareness: number;    // Нахождение в пределах карты
        total: number;           // Суммарная награда для головы движения
    };

    // Награды для головы прицеливания
    aim: {
        distance: number;        // Расстояние до цели
        accuracy: number;        // Точность прицеливания
        tracking: number;        // Активное отслеживание цели
        mapAwareness: number;    // Нахождение в пределах карты
        total: number;           // Суммарная награда для головы прицеливания
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
        shoot: { shootDecision: 0, total: 0 },
        movement: { speed: 0, enemiesPositioning: 0, bulletAvoidance: 0, mapAwareness: 0, total: 0 },
        aim: { accuracy: 0, tracking: 0, distance: 0, mapAwareness: 0, total: 0 },
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
    const [beforePredictTankX, beforePredictTankY] = TankInputTensor.position.getBatche(tankEid);
    // const [beforePredictTankSpeedX, beforePredictTankSpeedY] = TankInputTensor.speed.getBatche(tankEid);
    const [beforePredictTurretTargetX, beforePredictTurretTargetY] = TankInputTensor.turretTarget.getBatche(tankEid);
    const beforePredictEnemiesData = TankInputTensor.enemiesData.getBatche(tankEid);
    const beforePredictBulletsData = TankInputTensor.bulletsData.getBatche(tankEid);

    // -- current state --
    const currentHealth = getTankHealth(tankEid);
    const isShooting = TankController.shoot[tankEid] > 0;
    const [currentTankX, currentTankY] = RigidBodyState.position.getBatche(tankEid);
    const [currentTankSpeedX, currentTankSpeedY] = RigidBodyState.linvel.getBatche(tankEid);
    const [currentTurretTargetX, currentTurretTargetY] = getMatrixTranslation(LocalTransform.matrix.getBatche(Tank.aimEid[tankEid]));
    // const currentShootings = TankController.shoot[tankEid] > 0;
    const currentEnemies = Array.from(findTankEnemies(tankEid));
    const currentDangerBullets = Array.from(findTankDangerBullets(tankEid));

    // Инициализируем пустую структуру наград
    const rewards = initializeRewards();

    // 1. Награда за выживание
    rewards.common.health = currentHealth * REWARD_WEIGHTS.COMMON.HEALTH;
    rewards.common.survival = (step / CONFIG.maxFrames) * REWARD_WEIGHTS.COMMON.SURVIVAL;

    // 2. Расчет награды за нахождение в пределах карты
    rewards.movement.mapAwareness = calculateTankMapAwarenessReward(
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
    rewards.shoot.shootDecision = calculateShootingReward(
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
    rewards.movement.bulletAvoidance = bulletAvoidanceResult.reward;

    // 7. Награда за движение и позиционирование
    const movementRewardResult = calculateMovementReward(
        currentTankSpeedX,
        currentTankSpeedY,
        bulletAvoidanceResult.maxDangerLevel,
        aimingResult.hasTargets,
        aimingResult.closestEnemyDist,
    );

    rewards.common.total = rewards.common.health + rewards.common.survival;

    rewards.movement.speed = movementRewardResult.speed;
    rewards.movement.enemiesPositioning = movementRewardResult.positioning;

    // Рассчитываем итоговые значения
    rewards.aim.total = (rewards.aim.accuracy + rewards.aim.tracking + rewards.aim.distance + rewards.aim.mapAwareness);
    rewards.shoot.total = (rewards.shoot.shootDecision);
    rewards.movement.total = (rewards.movement.speed + rewards.movement.enemiesPositioning +
        rewards.movement.bulletAvoidance + rewards.movement.mapAwareness);

    // Общая итоговая награда
    rewards.totalReward = rewards.shoot.total + rewards.movement.total + rewards.aim.total;

    isVerboseLog() &&
    console.log(`[Reward] Tank ${ tankEid }
    aim: ${ rewards.aim.total }
    shoot: ${ rewards.shoot.total }
    move: ${ rewards.movement.total }
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
    enemiesList: number[],
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
    let hasTargets = enemiesList.length > 0;
    let closestEnemyDist = Number.MAX_VALUE;
    const turretTargetDistance = hypot(turretTargetX - tankX, turretTargetY - tankY);

    // Анализируем предыдущее прицеливание
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_ENEMIES; i++) {
        const enemyId = beforePredictEnemiesData[i * 6];
        const enemyX = beforePredictEnemiesData[i * 6 + 1];
        const enemyY = beforePredictEnemiesData[i * 6 + 2];

        if (enemyId === 0) continue;

        const prevAimQuality = evaluateAimQuality(
            tankX, tankY, prevTurretTargetX, prevTurretTargetY, enemyX, enemyY,
        );

        if (prevAimQuality > prevBestAimQuality) {
            prevBestAimQuality = prevAimQuality;
            prevBestAimTargetId = enemyId;
        }
    }

    // Анализируем всех видимых врагов для текущего состояния
    for (let i = 0; i < enemiesList.length; i++) {
        const enemyId = enemiesList[i];
        const enemyX = RigidBodyState.position.get(enemyId, 0);
        const enemyY = RigidBodyState.position.get(enemyId, 1);

        const distToEnemy = hypot(tankX - enemyX, tankY - enemyY);

        // Обновляем дистанцию до ближайшего врага
        if (distToEnemy < closestEnemyDist) {
            closestEnemyDist = distToEnemy;
        }

        const currentAimQuality = evaluateAimQuality(
            tankX, tankY, turretTargetX, turretTargetY, enemyX, enemyY,
        );

        // Отслеживаем лучшее прицеливание
        if (currentAimQuality > bestAimQuality) {
            bestAimQuality = currentAimQuality;
            bestAimTargetId = enemyId;
        }
    }

    // Награда за качество прицеливания и дистанцию до цели
    const aimQualityReward =
        bestAimQuality * REWARD_WEIGHTS.AIM.QUALITY
        + bestAimTargetId === 0 ? REWARD_WEIGHTS.AIM.NO_TARGET_PENALTY : 0;

    // Награда за дистанцию прицеливания
    const aimDistanceReward = bestAimQuality *
        (centerStep(0, 700, turretTargetDistance) * REWARD_WEIGHTS.AIM.DISTANCE
            + smoothstep(700, 1000, turretTargetDistance) * REWARD_WEIGHTS.AIM.DISTANCE_PENALTY);

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

// Вспомогательная функция для оценки качества прицеливания
function evaluateAimQuality(
    tankX: number,
    tankY: number,
    targetX: number,
    targetY: number,
    enemyX: number,
    enemyY: number,
): number {
    // Оценка точности прямого прицеливания
    const distFromTargetToEnemy = hypot(targetX - enemyX, targetY - enemyY);
    const directAimQuality = smoothstep(TANK_RADIUS * 1.5, TANK_RADIUS * 0.8, distFromTargetToEnemy);

    let combinedAimQuality = directAimQuality * 0.5; // По умолчанию только 50% от компонента точного попадания

    // Вычисляем вектора
    const vectorTankToEnemyX = enemyX - tankX;
    const vectorTankToEnemyY = enemyY - tankY;
    const vectorTankToTargetX = targetX - tankX;
    const vectorTankToTargetY = targetY - tankY;

    // Нормализуем вектора
    const tankToEnemyLength = hypot(vectorTankToEnemyX, vectorTankToEnemyY);
    const tankToTargetLength = hypot(vectorTankToTargetX, vectorTankToTargetY);


    if (tankToEnemyLength > 0 && tankToTargetLength > 0) {
        const normalizedTankToEnemyX = vectorTankToEnemyX / tankToEnemyLength;
        const normalizedTankToEnemyY = vectorTankToEnemyY / tankToEnemyLength;
        const normalizedTankToTargetX = vectorTankToTargetX / tankToTargetLength;
        const normalizedTankToTargetY = vectorTankToTargetY / tankToTargetLength;

        // Скалярное произведение нормализованных векторов
        const dotProduct = normalizedTankToEnemyX * normalizedTankToTargetX +
            normalizedTankToEnemyY * normalizedTankToTargetY;

        // Проверяем, пересекает ли вектор прицела танк
        if (dotProduct > 0) {
            // Вычисляем проекцию точки прицела на линию танк-враг
            const projectionFactor = (vectorTankToTargetX * normalizedTankToEnemyX +
                vectorTankToTargetY * normalizedTankToEnemyY) / tankToEnemyLength;

            const projectionX = tankX + normalizedTankToEnemyX * projectionFactor * tankToEnemyLength;
            const projectionY = tankY + normalizedTankToEnemyY * projectionFactor * tankToEnemyLength;

            // Расстояние от цели прицела до проекции на линию
            const distToLine = hypot(targetX - projectionX, targetY - projectionY);

            // Оцениваем качество совпадения с вектором
            const vectorAlignmentQuality = smoothstep(TANK_RADIUS * 1.5, TANK_RADIUS * 0.8, distToLine)
                * smoothstep(TANK_RADIUS * 4, TANK_RADIUS * 1.5, distFromTargetToEnemy);

            // Объединяем оба компонента награды
            combinedAimQuality += vectorAlignmentQuality * 0.5;
        }
    }

    return combinedAimQuality;
}

/**
 * Расчет награды за отслеживание целей
 */
function calculateTrackingReward(
    aimingResult: ReturnType<typeof analyzeAiming>,
): number {
    const { bestAimQuality, prevBestAimQuality } = aimingResult;

    // Вычисляем изменение комбинированного качества прицеливания
    const deltaAimQuality = bestAimQuality - prevBestAimQuality;

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
        shootingReward += bestAimQuality * REWARD_WEIGHTS.SHOOTING.AIMED;
    } else if (bestAimQuality > 0.5) {
        // Небольшой штраф за отсутствие стрельбы при хорошем прицеливании
        shootingReward += REWARD_WEIGHTS.SHOOTING.AIMED_PENALTY * smoothstep(0.8, 1.0, bestAimQuality);
    }

    return shootingReward;
}

/**
 * Расчет награды за движение
 */
function calculateMovementReward(
    tankSpeedX: number,
    tankSpeedY: number,
    maxDangerLevel: number,
    hasTargets: boolean,
    closestEnemyDist: number,
): { speed: number; positioning: number } {
    let speedReward = 0;
    let positioningReward = 0;

    // Базовая награда за движение (плавно растет с увеличением скорости)
    const tankSpeed = hypot(tankSpeedX, tankSpeedY);
    // const prevTankSpeed = hypot(prevTankSpeedX, prevTankSpeedY);
    const speedFactor = smoothstep(0, TANK_MAX_SPEED, tankSpeed);

    // Базовая награда за движение
    speedReward += speedFactor * REWARD_WEIGHTS.MOVEMENT.BASE;

    // Стратегическое движение при наличии опасности
    if (maxDangerLevel > 0) {
        // Дополнительная награда за движение при наличии опасных пуль
        speedReward += speedFactor * REWARD_WEIGHTS.MOVEMENT.STRATEGIC;
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
        const id = beforePredictBulletsData[i * 5];
        if (id === 0) continue;

        prevBullets.set(id, {
            x: beforePredictBulletsData[i * 5 + 1],
            y: beforePredictBulletsData[i * 5 + 2],
            vx: beforePredictBulletsData[i * 5 + 3],
            vy: beforePredictBulletsData[i * 5 + 4],
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