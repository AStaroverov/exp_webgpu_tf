import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';
import { RigidBodyState } from '../../ECS/Components/Physical.ts';
import { TankController } from '../../ECS/Components/TankController.ts';
import { findTankDangerBullets, findTankEnemies } from '../../ECS/Systems/createTankInputTensorSystem.ts';
import { centerStep, hypot, lerp, max, min, smoothstep } from '../../../../../lib/math.ts';
import { TANK_RADIUS } from './consts.ts';
import { isVerboseLog } from './utils.ts';

// Константы для калибровки вознаграждений
const REWARD_WEIGHTS = {
    // Основные компоненты наград
    HEALTH_CHANGE: 0.5,          // За потерю здоровья
    HEALTH_BONUS: 0.05,          // За поддержание здоровья
    SURVIVAL: 0.05,              // За выживание

    MOVEMENT_BASE: 0.1,          // За базовое движение
    STRATEGIC_MOVEMENT: 0.3,     // За стратегическое движение

    BULLET_AVOIDANCE: 0.4,       // За избегание пуль
    BULLET_AVOIDANCE_PENALTY: -0.4,       // За избегание пуль

    MAP_BORDER_PENALTY: -1.0,            // За выход за границы
    BORDER_GRADIENT_PENALTY: -0.5,       // За приближение к границе

    DISTANCE_KEEPING: 0.5,       // За поддержание дистанции
    DISTANCE_KEEPING_PENALTY: -0.5,       // За поддержание дистанции

    SHOOTING_AIMED: 1.0,         // За прицельную стрельбу
    SHOOTING_AIMED_PENALTY: -0.5, // Штраф за стрельбу в пустоту
    SHOOTING_RANDOM_PENALTY: -0.2,       // За беспорядочную стрельбу

    AIM_QUALITY: 1.0,            // За точное прицеливание
    AIM_TRACKING: 1.0,           // За активное отслеживание врага
    AIM_TRACKING_PENALTY: -0.5,           // За активное отслеживание врага
    AIM_DISTANCE_PENALTY: -0.5,          // За расстояние до цели
};

const REWARD_WEIGHTS_ORIGINAL = structuredClone(REWARD_WEIGHTS);

// Структура для хранения многокомпонентных наград
export interface ComponentRewards {
    // Награды для головы стрельбы
    shoot: {
        aimQuality: number;      // Качество прицеливания
        shootDecision: number;   // Решение о стрельбе
        total: number;           // Суммарная награда для головы стрельбы
    };

    // Награды для головы движения
    movement: {
        speed: number;           // Скорость движения
        positioning: number;     // Позиционирование
        avoidance: number;       // Избегание опасности
        mapAwareness: number;    // Нахождение в пределах карты
        total: number;           // Суммарная награда для головы движения
    };

    // Награды для головы прицеливания
    aim: {
        distance: number;        // Расстояние до цели
        accuracy: number;        // Точность прицеливания
        tracking: number;        // Активное отслеживание цели
        total: number;           // Суммарная награда для головы прицеливания
    };

    // Общие награды
    common: {
        health: number;          // Здоровье
        survival: number;        // Выживание
        total: number;           // Общая награда
    };

    // Общая суммарная награда
    totalReward: number;
}

/**
 * Инициализация пустой структуры наград
 */
function initializeRewards(): ComponentRewards {
    return {
        shoot: { aimQuality: 0, shootDecision: 0, total: 0 },
        movement: { speed: 0, positioning: 0, avoidance: 0, mapAwareness: 0, total: 0 },
        aim: { accuracy: 0, tracking: 0, distance: 0, total: 0 },
        common: { health: 0, survival: 0, total: 0 },
        totalReward: 0,
    };
}

const TRAIN_SECTION = 1000;

export function calculateReward(
    tankEid: number,
    actions: ArrayLike<number>,
    width: number,
    height: number,
    episode: number,
): ComponentRewards {
    Object.assign(REWARD_WEIGHTS, REWARD_WEIGHTS_ORIGINAL);

    const aimMultiplier = 1.0;
    const shootMultiplier = smoothstep(TRAIN_SECTION, TRAIN_SECTION * 2, episode);
    const moveMultiplier = smoothstep(TRAIN_SECTION * 3, TRAIN_SECTION * 4, episode);
    const penaltyMultiplier = smoothstep(TRAIN_SECTION * 5, TRAIN_SECTION * 6, episode);

    for (const penaltyKey in REWARD_WEIGHTS) {
        if (penaltyKey.endsWith('_PENALTY')) {
            // @ts-ignore
            REWARD_WEIGHTS[penaltyKey] *= penaltyMultiplier;
        }
    }

    // before predict
    // const beforePredictHealth = TankInputTensor.health[tankEid];
    const [beforePredictTankX, beforePredictTankY] = TankInputTensor.position.getBatche(tankEid);
    const [beforePredictTankSpeedX, beforePredictTankSpeedY] = TankInputTensor.speed.getBatche(tankEid);
    const [beforePredictTurretTargetX, beforePredictTurretTargetY] = TankInputTensor.turretTarget.getBatche(tankEid);
    const beforePredictEnemiesData = TankInputTensor.enemiesData.getBatche(tankEid);
    const beforePredictBulletsData = TankInputTensor.bulletsData.getBatche(tankEid);

    // current state
    // const currentHealth = getTankHealth(tankEid);
    const [currentTankX, currentTankY] = RigidBodyState.position.getBatche(tankEid);
    const [currentTankSpeedX, currentTankSpeedY] = RigidBodyState.linvel.getBatche(tankEid);
    const [currentTurretTargetX, currentTurretTargetY] = TankController.turretTarget.getBatche(tankEid);
    // const currentShootings = TankController.shoot[tankEid] > 0;
    const [currentEnemiesCount, currentEnemiesList] = findTankEnemies(tankEid);
    const [currentDangerBulletsCount, currentDangerBulletsList] = findTankDangerBullets(tankEid);

    const isShooting = actions[0] > 0.5; // выстрел
    const actionMoveDir = actions[1] || 0; // forward/backward
    const actionMoveRot = actions[2] || 0; // rotate left/right
    // const actionAimDeltaX = actions[3] || 0;  // change aim by X[-1,1], from beforePredictTurretTargetXY.x
    // const actionAimDeltaY = actions[4] || 0;  // change aim by Y[-1,1], from beforePredictTurretTargetXY.y

    // Инициализируем пустую структуру наград
    const rewards = initializeRewards();

    // 1. Расчет награды за здоровье и выживание
    // rewards.common.health = calculateHealthReward(currentHealth, beforePredictHealth);
    // rewards.common.survival = calculateSurvivalReward(currentHealth);

    // 2. Расчет награды за нахождение в пределах карты
    rewards.movement.mapAwareness = calculateMapReward(currentTankX, currentTankY, width, height);

    // 3. Анализ целей и вычисление награды за прицеливание
    const aimingResult = analyzeAiming(
        currentTankX,
        currentTankY,
        currentTurretTargetX,
        currentTurretTargetY,
        currentEnemiesCount,
        currentEnemiesList,
        beforePredictTurretTargetX,
        beforePredictTurretTargetY,
        beforePredictEnemiesData,
    );

    rewards.aim.accuracy = aimingResult.aimQualityReward;
    rewards.aim.distance = aimingResult.aimDistanceReward;
    rewards.shoot.aimQuality = aimingResult.aimQualityReward;

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
        currentDangerBulletsCount,
        currentDangerBulletsList,
        beforePredictBulletsData,
    );
    rewards.movement.avoidance = bulletAvoidanceResult.reward;

    // 7. Награда за движение и позиционирование
    const movementRewardResult = calculateMovementReward(
        currentTankSpeedX,
        currentTankSpeedY,
        beforePredictTankSpeedX,
        beforePredictTankSpeedY,
        actionMoveDir,
        actionMoveRot,
        bulletAvoidanceResult.maxDangerLevel,
        aimingResult.hasTargets,
        aimingResult.closestEnemyDist,
    );

    rewards.movement.speed = movementRewardResult.speed;
    rewards.movement.positioning = movementRewardResult.positioning;

    // Рассчитываем итоговые значения
    rewards.aim.total = aimMultiplier * (rewards.aim.accuracy + rewards.aim.tracking + rewards.aim.distance);
    rewards.shoot.total = shootMultiplier * (rewards.shoot.aimQuality + rewards.shoot.shootDecision);
    rewards.movement.total = moveMultiplier * (rewards.movement.speed + rewards.movement.positioning +
        rewards.movement.avoidance + rewards.movement.mapAwareness);
    rewards.common.total = rewards.common.health + rewards.common.survival;

    // Общая итоговая награда
    rewards.totalReward = rewards.shoot.total + rewards.movement.total +
        rewards.aim.total + rewards.common.total;

    isVerboseLog() &&
    console.log(`>> Tank ${ tankEid }
    aim: ${ rewards.aim.total }
    shoot: ${ rewards.shoot.total }
    move: ${ rewards.movement.total }
    common: ${ rewards.common.total }
    total: ${ rewards.totalReward }
    `);

    return rewards;
}

/**
 * Расчет награды за нахождение в пределах карты
 */
function calculateMapReward(
    entityX: number,
    entityY: number,
    width: number,
    height: number,
): number {
    if (entityX >= 0 && entityX <= width && entityY >= 0 && entityY <= height) {
        // В пределах карты - используем плавный градиент
        const borderDistance = min(
            entityX,
            entityY,
            width - entityX,
            height - entityY,
        );
        // Плавное уменьшение награды при приближении к границе
        const borderFactor = 1 - smoothstep(0, 50, borderDistance);

        return REWARD_WEIGHTS.BORDER_GRADIENT_PENALTY * borderFactor;
    } else {
        // Вышел за границы карты
        return REWARD_WEIGHTS.MAP_BORDER_PENALTY;
    }
}

/**
 * Расчет награды за сохранение здоровья
 */
function calculateHealthReward(
    currentHealth: number,
    prevHealth: number,
): number {
    // Награда за изменение здоровья
    const healthChange = currentHealth - prevHealth;
    const healthChangeReward = healthChange * REWARD_WEIGHTS.HEALTH_CHANGE;

    // Бонус за текущее здоровье
    const healthBonusReward = currentHealth * REWARD_WEIGHTS.HEALTH_BONUS;

    return healthChangeReward + healthBonusReward;
}

/**
 * Расчет награды за выживание
 */
function calculateSurvivalReward(currentHealth: number): number {
    let survivalReward = REWARD_WEIGHTS.SURVIVAL; // Базовый бонус за выживание

    // Дополнительный бонус при низком здоровье (плавно растет по мере снижения здоровья)
    if (currentHealth < 0.5) {
        const lowHealthFactor = smoothstep(0.5, 0, currentHealth);
        survivalReward += lowHealthFactor * REWARD_WEIGHTS.SURVIVAL;
    }

    return survivalReward;
}

/**
 * Анализ прицеливания и видимых врагов
 */
function analyzeAiming(
    tankX: number,
    tankY: number,
    turretTargetX: number,
    turretTargetY: number,
    enemiesCount: number,
    enemiesList: Float64Array,
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
    let hasTargets = enemiesCount > 0;
    let closestEnemyDist = Number.MAX_VALUE;
    const turretTargetDistance = hypot(turretTargetX - tankX, turretTargetY - tankY);

    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_ENEMIES; i++) {
        const enemyId = beforePredictEnemiesData[i * 6];
        const enemyX = beforePredictEnemiesData[i * 6 + 1];
        const enemyY = beforePredictEnemiesData[i * 6 + 2];

        if (enemyId === 0) continue;

        // Оценка точности предыдущего прицеливания
        const prevDistFromTargetToEnemy = hypot(prevTurretTargetX - enemyX, prevTurretTargetY - enemyY);
        const prevDirectAimQuality = smoothstep(TANK_RADIUS * 1.5, TANK_RADIUS * 0.8, prevDistFromTargetToEnemy);

        // Оценка совпадения предыдущего прицела с вектором между танком и врагом
        const prevVectorTankToEnemyX = enemyX - tankX;
        const prevVectorTankToEnemyY = enemyY - tankY;
        const prevVectorTankToTargetX = prevTurretTargetX - tankX;
        const prevVectorTankToTargetY = prevTurretTargetY - tankY;

        // Нормализуем предыдущие вектора
        const prevTankToEnemyLength = hypot(prevVectorTankToEnemyX, prevVectorTankToEnemyY);
        const prevTankToTargetLength = hypot(prevVectorTankToTargetX, prevVectorTankToTargetY);

        let prevCombinedAimQuality = prevDirectAimQuality * 0.5; // По умолчанию только 50% от компонента точного попадания

        if (prevTankToEnemyLength > 0 && prevTankToTargetLength > 0) {
            const prevNormalizedTankToEnemyX = prevVectorTankToEnemyX / prevTankToEnemyLength;
            const prevNormalizedTankToEnemyY = prevVectorTankToEnemyY / prevTankToEnemyLength;
            const prevNormalizedTankToTargetX = prevVectorTankToTargetX / prevTankToTargetLength;
            const prevNormalizedTankToTargetY = prevVectorTankToTargetY / prevTankToTargetLength;

            // Скалярное произведение нормализованных предыдущих векторов
            const prevDotProduct = prevNormalizedTankToEnemyX * prevNormalizedTankToTargetX +
                prevNormalizedTankToEnemyY * prevNormalizedTankToTargetY;

            // Преобразуем в диапазон [0, 1]
            const prevVectorAlignmentQuality = (prevDotProduct + 1) / 2;

            // Объединяем оба компонента предыдущей награды
            prevCombinedAimQuality = prevDirectAimQuality * 0.5 + prevVectorAlignmentQuality * 0.5;
        }

        if (prevCombinedAimQuality > prevBestAimQuality) {
            prevBestAimQuality = prevCombinedAimQuality;
            prevBestAimTargetId = enemyId;
        }
    }

    // Анализируем всех видимых врагов для текущего состояния
    for (let i = 0; i < enemiesCount; i++) {
        const enemyId = enemiesList[i];
        const enemyX = RigidBodyState.position.get(enemyId, 0);
        const enemyY = RigidBodyState.position.get(enemyId, 1);

        const distToEnemy = hypot(tankX - enemyX, tankY - enemyY);

        // Обновляем дистанцию до ближайшего врага
        if (distToEnemy < closestEnemyDist) {
            closestEnemyDist = distToEnemy;
        }

        // Оценка точности прицеливания с плавным переходом (50% от общей оценки)
        const distFromTargetToEnemy = hypot(turretTargetX - enemyX, turretTargetY - enemyY);
        const directAimQuality = smoothstep(TANK_RADIUS * 1.5, TANK_RADIUS * 0.8, distFromTargetToEnemy);

        // Оценка совпадения прицела с вектором между танком и врагом (50% от общей оценки)
        // Вычисляем проекцию вектора прицела на вектор к врагу
        const vectorTankToEnemyX = enemyX - tankX;
        const vectorTankToEnemyY = enemyY - tankY;
        const vectorTankToTargetX = turretTargetX - tankX;
        const vectorTankToTargetY = turretTargetY - tankY;

        // Нормализуем вектора
        const tankToEnemyLength = hypot(vectorTankToEnemyX, vectorTankToEnemyY);
        const tankToTargetLength = hypot(vectorTankToTargetX, vectorTankToTargetY);

        if (tankToEnemyLength > 0 && tankToTargetLength > 0) {
            const normalizedTankToEnemyX = vectorTankToEnemyX / tankToEnemyLength;
            const normalizedTankToEnemyY = vectorTankToEnemyY / tankToEnemyLength;
            const normalizedTankToTargetX = vectorTankToTargetX / tankToTargetLength;
            const normalizedTankToTargetY = vectorTankToTargetY / tankToTargetLength;

            // Скалярное произведение нормализованных векторов (косинус угла между ними)
            const dotProduct = normalizedTankToEnemyX * normalizedTankToTargetX +
                normalizedTankToEnemyY * normalizedTankToTargetY;

            // Преобразуем в диапазон [0, 1], где 1 это полное совпадение направлений
            const vectorAlignmentQuality = (dotProduct + 1) / 2;

            // Объединяем оба компонента награды (по 50% каждый)
            const combinedAimQuality = directAimQuality * 0.5 + vectorAlignmentQuality * 0.5;

            // Отслеживаем лучшее прицеливание
            if (combinedAimQuality > bestAimQuality) {
                bestAimQuality = combinedAimQuality;
                bestAimTargetId = enemyId;
            }
        } else {
            // Если нельзя вычислить совпадение векторов, используем только прямое прицеливание
            if (directAimQuality > bestAimQuality) {
                bestAimQuality = directAimQuality * 0.5; // только 50% от полной награды
                bestAimTargetId = enemyId;
            }
        }
    }

    // Награда за качество прицеливания и дистанцию до цели
    const aimQualityReward = bestAimQuality * REWARD_WEIGHTS.AIM_QUALITY;

    // Награда за дистанцию прицеливания - штраф за слишком далекое прицеливание
    const aimDistanceReward = hasTargets
        ? smoothstep(600, 1000, turretTargetDistance) * REWARD_WEIGHTS.AIM_DISTANCE_PENALTY
        : 0;

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

/**
 * Расчет награды за отслеживание целей
 */
function calculateTrackingReward(
    aimingResult: ReturnType<typeof analyzeAiming>,
): number {
    const { bestAimTargetId, prevBestAimTargetId, bestAimQuality, prevBestAimQuality } = aimingResult;
    let trackingReward = 0;

    // Награда за поддержание прицела на том же враге
    if (bestAimTargetId !== 0 && bestAimTargetId === prevBestAimTargetId) {
        // Вычисляем изменение комбинированного качества прицеливания
        const deltaAimQuality = bestAimQuality - prevBestAimQuality;

        // Базовая награда за отслеживание той же цели
        const baseTrackingReward = 0.2 * REWARD_WEIGHTS.AIM_TRACKING;

        // Дополнительная награда за улучшение качества прицеливания
        const improvementReward = deltaAimQuality > 0
            ? deltaAimQuality * REWARD_WEIGHTS.AIM_TRACKING
            : 0;

        // Небольшой штраф за ухудшение прицеливания, но не такой сильный как награда за улучшение
        const deteriorationPenalty = deltaAimQuality < 0
            ? deltaAimQuality * 0.5 * REWARD_WEIGHTS.AIM_TRACKING_PENALTY
            : 0;

        trackingReward += baseTrackingReward + improvementReward + deteriorationPenalty;
    }
    // Штраф за смену цели
    else if (bestAimTargetId !== 0 && prevBestAimTargetId !== 0 && bestAimTargetId !== prevBestAimTargetId) {
        // Базовый штраф за смену цели
        trackingReward += 0.1 * REWARD_WEIGHTS.AIM_TRACKING_PENALTY;

        // Но если новая цель имеет значительно лучшее качество прицеливания, уменьшаем штраф
        if (bestAimQuality > prevBestAimQuality * 1.2) { // Если новая цель на 20% лучше
            const qualityImprovement = bestAimQuality - prevBestAimQuality;
            trackingReward += qualityImprovement * 0.3 * REWARD_WEIGHTS.AIM_TRACKING; // Компенсируем часть штрафа
        }
    }
    // Штраф за потерю цели
    else if (prevBestAimTargetId !== 0 && bestAimTargetId === 0) {
        // Штраф зависит от того, насколько хорошо мы целились раньше
        // Потеря хорошей цели дает больший штраф
        const lossQualityFactor = Math.min(1, prevBestAimQuality * 2); // Коэффициент от 0 до 1
        trackingReward += 0.3 * lossQualityFactor * REWARD_WEIGHTS.AIM_TRACKING_PENALTY;
    }
    // Награда за обнаружение новой цели, если раньше целей не было
    else if (prevBestAimTargetId === 0 && bestAimTargetId !== 0) {
        // Награда зависит от качества прицеливания на новой цели
        trackingReward += 0.5 * bestAimQuality * REWARD_WEIGHTS.AIM_TRACKING;
    }

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
        if (bestAimQuality < 0.05) {
            // Штраф за стрельбу при плохом прицеливании (плавно уменьшается с ростом точности)
            shootingReward += (1 - smoothstep(0, 0.3, bestAimQuality)) * REWARD_WEIGHTS.SHOOTING_RANDOM_PENALTY;
        } else {
            // Плавная награда за стрельбу в зависимости от точности прицеливания
            shootingReward += bestAimQuality * REWARD_WEIGHTS.SHOOTING_AIMED;
        }
    } else if (bestAimQuality > 0.8) {
        // Небольшой штраф за отсутствие стрельбы при хорошем прицеливании
        shootingReward += REWARD_WEIGHTS.SHOOTING_AIMED_PENALTY * 0.3 * smoothstep(0.8, 1.0, bestAimQuality);
    }

    return shootingReward;
}

/**
 * Расчет награды за движение
 */
function calculateMovementReward(
    tankSpeedX: number,
    tankSpeedY: number,
    _prevTankSpeedX: number,
    _prevTankSpeedY: number,
    _moveDirAction: number,
    _moveRotAction: number,
    maxDangerLevel: number,
    hasTargets: boolean,
    closestEnemyDist: number,
): { speed: number; positioning: number } {
    let speedReward = 0;
    let positioningReward = 0;

    // Базовая награда за движение (плавно растет с увеличением скорости)
    const tankSpeed = hypot(tankSpeedX, tankSpeedY);
    // const prevTankSpeed = hypot(prevTankSpeedX, prevTankSpeedY);
    const speedFactor = smoothstep(0, 300, tankSpeed);

    // Оценка изменения скорости (разгон или торможение)
    // const speedChange = tankSpeed - prevTankSpeed;
    // const isAccelerating = speedChange > 0;

    // Базовая награда за движение
    speedReward += speedFactor * REWARD_WEIGHTS.MOVEMENT_BASE;

    // Дополнительная награда за успешное управление
    // if (isAccelerating && Math.abs(moveDirAction) > 0.1) {
    //     // Награда за успешное ускорение
    //     speedReward += 0.1 * REWARD_WEIGHTS.MOVEMENT_BASE * Math.abs(moveDirAction);
    // }
    //
    // // Награда за маневрирование (поворот)
    // if (Math.abs(moveRotAction) > 0.1) {
    //     speedReward += 0.1 * REWARD_WEIGHTS.MOVEMENT_BASE * Math.abs(moveRotAction);
    // }

    // Стратегическое движение при наличии опасности
    if (maxDangerLevel > 0.3) {
        // Дополнительная награда за движение при наличии опасных пуль
        speedReward += smoothstep(0.3, 0.6, maxDangerLevel) * REWARD_WEIGHTS.STRATEGIC_MOVEMENT;
    }

    // Награда за позиционирование относительно врагов
    if (hasTargets) {
        if (closestEnemyDist < 3 * TANK_RADIUS) {
            // Штраф за слишком близкое расстояние
            const tooClosePenalty = 1 - smoothstep(0, 3 * TANK_RADIUS, closestEnemyDist);
            positioningReward += tooClosePenalty * REWARD_WEIGHTS.DISTANCE_KEEPING_PENALTY;
        } else if (closestEnemyDist <= 600) {
            // Награда за оптимальную дистанцию
            const optimalDistanceReward = lerp(0.3, 1, centerStep(3 * TANK_RADIUS, 600, closestEnemyDist));
            positioningReward += optimalDistanceReward * REWARD_WEIGHTS.DISTANCE_KEEPING;
        } else {
            // Мягкий штраф за слишком большую дистанцию
            const tooFarPenalty = smoothstep(600, 800, closestEnemyDist) * REWARD_WEIGHTS.DISTANCE_KEEPING_PENALTY / 5;
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
    dangerBulletsCount: number,
    dangerBulletsList: Float64Array,
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
    for (let i = 0; i < dangerBulletsCount; i++) {
        const bulletId = dangerBulletsList[i];
        const bulletX = RigidBodyState.position.get(bulletId, 0);
        const bulletY = RigidBodyState.position.get(bulletId, 1);
        const bulletVx = RigidBodyState.linvel.get(bulletId, 0);
        const bulletVy = RigidBodyState.linvel.get(bulletId, 1);

        // Находим информацию о пуле в предыдущем состоянии
        const prevBulletInfo = prevBullets.get(bulletId);

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
    prevBulletInfo?: { x: number, y: number, vx: number, vy: number },
): { reward: number; isDangerous: boolean; dangerLevel: number; wasAvoided: boolean } {
    // Результаты анализа
    let reward = 0;
    let isDangerous = false;
    let dangerLevel = 0;
    let wasAvoided = false;

    // Анализируем текущую опасность пули
    const bulletSpeed = hypot(bulletVx, bulletVy);
    if (bulletSpeed < 100) return { reward, isDangerous, dangerLevel, wasAvoided };

    const bulletDirectionX = bulletVx / bulletSpeed;
    const bulletDirectionY = bulletVy / bulletSpeed;

    // Вектор от пули к танку
    const toTankX = tankX - bulletX;
    const toTankY = tankY - bulletY;

    // Определяем, движется ли пуля к танку
    const dotProduct = toTankX * bulletDirectionX + toTankY * bulletDirectionY;

    // Если пуля движется к танку
    if (dotProduct > 0) {
        // Проекция вектора на направление пули
        const projLength = dotProduct;

        // Точка ближайшего прохождения пули к танку
        const closestPointX = bulletX + bulletDirectionX * projLength;
        const closestPointY = bulletY + bulletDirectionY * projLength;

        // Расстояние в точке наибольшего сближения
        const minDist = hypot(closestPointX - tankX, closestPointY - tankY);

        // Если есть информация о предыдущем состоянии пули, проверим было ли уклонение
        if (prevBulletInfo) {
            const prevBulletSpeed = hypot(prevBulletInfo.vx, prevBulletInfo.vy);

            if (prevBulletSpeed > 100) {
                const prevBulletDirX = prevBulletInfo.vx / prevBulletSpeed;
                const prevBulletDirY = prevBulletInfo.vy / prevBulletSpeed;

                // Вектор от предыдущей позиции пули к предыдущей позиции танка
                const prevToTankX = prevTankX - prevBulletInfo.x;
                const prevToTankY = prevTankY - prevBulletInfo.y;

                // Проверяем, двигалась ли пуля к танку в предыдущем состоянии
                const prevDotProduct = prevToTankX * prevBulletDirX + prevToTankY * prevBulletDirY;

                if (prevDotProduct > 0) {
                    // Проекция вектора на направление пули в предыдущем состоянии
                    const prevProjLength = prevDotProduct;

                    // Точка ближайшего прохождения пули к танку в предыдущем состоянии
                    const prevClosestPointX = prevBulletInfo.x + prevBulletDirX * prevProjLength;
                    const prevClosestPointY = prevBulletInfo.y + prevBulletDirY * prevProjLength;

                    // Расстояние в точке наибольшего сближения в предыдущем состоянии
                    const prevMinDist = hypot(prevClosestPointX - prevTankX, prevClosestPointY - prevTankY);

                    // Если сближение стало меньше, значит танк приблизился к пуле - это плохо
                    // Если сближение стало больше, значит танк уклонился от пули - это хорошо
                    if (minDist > prevMinDist + 10) { // Добавляем небольшой порог для уверенности
                        wasAvoided = true;

                        // Награда за уклонение пропорциональна тому, насколько близко была пуля
                        const avoidanceQuality = smoothstep(TANK_RADIUS + 100, TANK_RADIUS, prevMinDist);
                        reward += avoidanceQuality * REWARD_WEIGHTS.BULLET_AVOIDANCE * 0.8;
                    }
                }
            }
        }

        // Оценка текущей опасности пули с плавным переходом
        if (minDist < 120) { // Увеличенное расстояние обнаружения для плавного перехода
            // Время до точки сближения
            const timeToClosest = projLength / bulletSpeed;

            // Плавная оценка опасности
            if (timeToClosest < 1.2) {
                // Используем smoothstep для плавного изменения опасности
                const distanceFactor = smoothstep(120, 40, minDist); // От 0 до 1 при приближении
                const timeFactor = smoothstep(1.2, 0.1, timeToClosest); // От 0 до 1 при приближении

                dangerLevel = distanceFactor * timeFactor;
                isDangerous = true;

                // Штраф пропорционален уровню опасности (уменьшаем, если было уклонение)
                reward += dangerLevel * REWARD_WEIGHTS.BULLET_AVOIDANCE_PENALTY * (wasAvoided ? 0.5 : 1.0);
            }
        } else if (!wasAvoided) {
            // Пуля пролетит далеко от танка - небольшая позитивная награда
            const avoidFactor = lerp(0.3, 0.8, smoothstep(200, 120, minDist));
            reward += avoidFactor * REWARD_WEIGHTS.BULLET_AVOIDANCE * 0.2;
        }
    }
    // else if (!wasAvoided) {
    //     // Пуля движется от танка - еще меньшая награда
    //     reward += 0.1 * REWARD_WEIGHTS.BULLET_AVOIDANCE;
    // }

    return { reward, isDangerous, dangerLevel, wasAvoided };
}