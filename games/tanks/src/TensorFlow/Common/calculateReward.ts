import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';
import { RigidBodyState } from '../../ECS/Components/Physical.ts';
import { findTankDangerBullets, findTankEnemies } from '../../ECS/Systems/createTankInputTensorSystem.ts';
import { abs, centerStep, hypot, lerp, max, min, smoothstep } from '../../../../../lib/math.ts';
import { TANK_RADIUS } from './consts.ts';
import { Actions, isVerboseLog, readAction } from './utils.ts';
import { getMatrixTranslation, LocalTransform } from '../../../../../src/ECS/Components/Transform.ts';
import { Tank } from '../../ECS/Components/Tank.ts';

// Константы для калибровки вознаграждений
const REWARD_GROUPS = [
    'AIM',
    'SHOOTING',
    'MAP_BORDER',
    'DISTANCE_KEEPING',
    'MOVEMENT',
    'BULLET_AVOIDANCE',
] as const;
let REWARD_WEIGHTS = {
    AIM: {
        QUALITY: 1.0,       // За точное прицеливание
        TRACKING: 1.0,      // За активное отслеживание врага
        DISTANCE: 1.0,      // За расстояние до цели
        NO_TARGET_PENALTY: -0.1, // За отсутствие целей
        TRACKING_PENALTY: -0.2, // За активное отслеживание врага
        DISTANCE_PENALTY: -0.2, // За расстояние до цели
    },

    MAP_BORDER: {
        BASE: 0.2,          // За нахождение в пределах карты
        RETURN: 1.0,          // За нахождение в пределах карты
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

    BULLET_AVOIDANCE: {
        BASE: 0.4,          // За избегание пуль
        PENALTY: -0.2,      // За избегание пуль
    },

    MOVEMENT: {
        BASE: 0.1,          // За базовое движение
        STRATEGIC: 0.3,     // За стратегическое движение
    },

    HEALTH_CHANGE: 0.5,          // За потерю здоровья
    HEALTH_BONUS: 0.05,          // За поддержание здоровья
    SURVIVAL: 0.05,              // За выживание
};

const REWARD_WEIGHTS_ORIGINAL = structuredClone(REWARD_WEIGHTS);

// Структура для хранения многокомпонентных наград
export interface ComponentRewards {
    // Награды для головы стрельбы
    shoot: {
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

    // Общая суммарная награда
    totalReward: number;
}

/**
 * Инициализация пустой структуры наград
 */
function initializeRewards(): ComponentRewards {
    return {
        shoot: { shootDecision: 0, total: 0 },
        movement: { speed: 0, positioning: 0, avoidance: 0, mapAwareness: 0, total: 0 },
        aim: { accuracy: 0, tracking: 0, distance: 0, total: 0 },
        totalReward: 0,
    };
}

const TRAIN_SECTION = 1_000;
let lastEpisode = 0;

function updateRewardWeights(episode: number) {
    if (episode === lastEpisode) return;
    REWARD_WEIGHTS = structuredClone(REWARD_WEIGHTS_ORIGINAL);

    let stp = -1;
    for (const group of REWARD_GROUPS) {
        const rewardMultiplier = smoothstep(TRAIN_SECTION * stp, TRAIN_SECTION * (stp + 1), episode);
        const penaltyMultiplier = smoothstep(TRAIN_SECTION * stp, TRAIN_SECTION * (stp + 1), episode);
        // @ts-ignore
        for (const key in REWARD_WEIGHTS[group]) {
            // @ts-ignore
            REWARD_WEIGHTS[group][key] *= key.endsWith('_PENALTY') ? penaltyMultiplier : rewardMultiplier;
        }
        if (group === 'AIM') continue;
        if (group === 'SHOOTING') continue;
        stp += 2;
    }

    lastEpisode = episode;
}

export function calculateReward(
    tankEid: number,
    actions: Actions,
    width: number,
    height: number,
    episode: number,
): ComponentRewards {
    updateRewardWeights(episode);

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
    const [currentTurretTargetX, currentTurretTargetY] = getMatrixTranslation(LocalTransform.matrix.getBatche(Tank.aimEid[tankEid]));
    // const currentShootings = TankController.shoot[tankEid] > 0;
    const [currentEnemiesCount, currentEnemiesList] = findTankEnemies(tankEid);
    const [currentDangerBulletsCount, currentDangerBulletsList] = findTankDangerBullets(tankEid);

    const {
        shoot: isShooting,
        move: actionMoveDir,
        rotate: actionMoveRot,
        // aim: [actionAimDeltaX, actionAimDeltaY],
    } = readAction(actions);

    // Инициализируем пустую структуру наград
    const rewards = initializeRewards();

    // 1. Расчет награды за здоровье и выживание
    // rewards.common.health = calculateHealthReward(currentHealth, beforePredictHealth);
    // rewards.common.survival = calculateSurvivalReward(currentHealth);

    // 2. Расчет награды за нахождение в пределах карты
    rewards.movement.mapAwareness = calculateMapReward(
        beforePredictTankX,
        beforePredictTankY,
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
        currentEnemiesCount,
        currentEnemiesList,
        beforePredictTurretTargetX,
        beforePredictTurretTargetY,
        beforePredictEnemiesData,
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
    rewards.aim.total = (rewards.aim.accuracy + rewards.aim.tracking + rewards.aim.distance);
    rewards.shoot.total = (rewards.shoot.shootDecision);
    rewards.movement.total = (rewards.movement.speed + rewards.movement.positioning +
        rewards.movement.avoidance + rewards.movement.mapAwareness);

    // Общая итоговая награда
    rewards.totalReward = rewards.shoot.total + rewards.movement.total + rewards.aim.total;

    isVerboseLog() &&
    console.log(`>> Tank ${ tankEid }
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
function calculateMapReward(
    prevX: number,
    prevY: number,
    x: number,
    y: number,
    width: number,
    height: number,
): number {
    const inMap = x >= 0 && x <= width && y >= 0 && y <= height;

    if (inMap) {
        const borderDistance = min(
            x,
            y,
            width - x,
            height - y,
        );
        const prevInMap = prevX >= 0 && prevX <= width && prevY >= 0 && prevY <= height;

        // Базовая награда за нахождение в пределах карты
        return REWARD_WEIGHTS.MAP_BORDER.BASE
            + (prevInMap ? 0 : 1) * REWARD_WEIGHTS.MAP_BORDER.RETURN
            // Штраф за приближение к границе
            + REWARD_WEIGHTS.MAP_BORDER.PENALTY * (1 - smoothstep(0, 50, borderDistance));
    } else {
        // Вышел за границы карты
        return REWARD_WEIGHTS.MAP_BORDER.PENALTY;
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
    for (let i = 0; i < enemiesCount; i++) {
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
    const aimDistanceReward =
        centerStep(0, 700, turretTargetDistance) * REWARD_WEIGHTS.AIM.DISTANCE
        + smoothstep(700, 1000, turretTargetDistance) * REWARD_WEIGHTS.AIM.DISTANCE_PENALTY;

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
    speedReward += speedFactor * REWARD_WEIGHTS.MOVEMENT.BASE;

    // Дополнительная награда за успешное управление
    // if (isAccelerating && Math.abs(moveDirAction) > 0.1) {
    //     // Награда за успешное ускорение
    //     speedReward += 0.1 * REWARD_WEIGHTS.MOVEMENT.BASE * Math.abs(moveDirAction);
    // }
    //
    // // Награда за маневрирование (поворот)
    // if (Math.abs(moveRotAction) > 0.1) {
    //     speedReward += 0.1 * REWARD_WEIGHTS.MOVEMENT.BASE * Math.abs(moveRotAction);
    // }

    // Стратегическое движение при наличии опасности
    if (maxDangerLevel > 0.3) {
        // Дополнительная награда за движение при наличии опасных пуль
        speedReward += smoothstep(0.3, 0.6, maxDangerLevel) * REWARD_WEIGHTS.MOVEMENT.STRATEGIC;
    }

    // Награда за позиционирование относительно врагов
    if (hasTargets) {
        if (closestEnemyDist < 3 * TANK_RADIUS) {
            // Штраф за слишком близкое расстояние
            const tooClosePenalty = 1 - smoothstep(0, 3 * TANK_RADIUS, closestEnemyDist);
            positioningReward += tooClosePenalty * REWARD_WEIGHTS.DISTANCE_KEEPING.PENALTY;
        } else if (closestEnemyDist <= 600) {
            // Награда за оптимальную дистанцию
            const optimalDistanceReward = lerp(0.3, 1, centerStep(3 * TANK_RADIUS, 600, closestEnemyDist));
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
                        reward += avoidanceQuality * REWARD_WEIGHTS.BULLET_AVOIDANCE.BASE * 0.8;
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
                reward += dangerLevel * REWARD_WEIGHTS.BULLET_AVOIDANCE.PENALTY * (wasAvoided ? 0.5 : 1.0);
            }
        } else if (!wasAvoided) {
            // Пуля пролетит далеко от танка - небольшая позитивная награда
            const avoidFactor = lerp(0.3, 0.8, smoothstep(200, 120, minDist));
            reward += avoidFactor * REWARD_WEIGHTS.BULLET_AVOIDANCE.BASE * 0.2;
        }
    }
    // else if (!wasAvoided) {
    //     // Пуля движется от танка - еще меньшая награда
    //     reward += 0.1 * REWARD_WEIGHTS.BULLET_AVOIDANCE;
    // }

    return { reward, isDangerous, dangerLevel, wasAvoided };
}