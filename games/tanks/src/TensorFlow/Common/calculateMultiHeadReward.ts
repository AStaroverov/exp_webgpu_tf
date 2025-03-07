import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';
import { TankController } from '../../ECS/Components/TankController.ts';
import { hypot, smoothstep } from '../../../../../lib/math.ts';
import { TANK_RADIUS } from './consts.ts';
import { isVerboseLog } from './utils.ts';

// Константы для калибровки вознаграждений
export const REWARD_WEIGHTS = {
    // Основные компоненты наград
    HEALTH_CHANGE: 0.5,          // За потерю здоровья
    HEALTH_BONUS: 0.05,          // За поддержание здоровья
    SURVIVAL: 0.05,              // За выживание

    MOVEMENT_BASE: 0.1,          // За базовое движение
    BULLET_AVOIDANCE: -2.0,      // За попадание под пули
    STRATEGIC_MOVEMENT: 0.3,     // За стратегическое движение

    MAP_BORDER: -2.0,            // За выход за границы
    BORDER_GRADIENT: -0.8,       // За приближение к границе
    DISTANCE_KEEPING: 0.5,       // За поддержание дистанции

    SHOOTING_AIMED: 1.2,         // За прицельную стрельбу
    SHOOTING_RANDOM: -0.7,       // За беспорядочную стрельбу

    AIM_QUALITY: 1.5,            // За точное прицеливание
    AIM_TRACKING: 2.0,           // За активное отслеживание врага
    AIM_STATIC_PENALTY: -0.2,    // Штраф за неподвижный прицел
};

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

// Информация о пулях для отслеживания
export type BulletInfo = {
    id: number;
    x: number;
    y: number;
    vx: number;
    vy: number;
    timestamp: number;
    wasAvoided: boolean;
};

// Хранение информации о пулях для каждого танка
const bulletMemory = new Map<number, Map<number, BulletInfo>>();

// Хранение предыдущих действий и состояний для анализа изменений
const previousTurretTargets = new Map<number, [number, number]>();
const previousEnemyPositions = new Map<number, Map<number, [number, number]>>();

/**
 * Расчет многокомпонентной награды для танка с плавными переходами
 */
export function calculateMultiHeadReward(
    tankEid: number,
    actions: ArrayLike<number>,
    prevHealth: number,
    width: number,
    height: number,
): ComponentRewards {
    const tankX = TankInputTensor.x[tankEid];
    const tankY = TankInputTensor.y[tankEid];
    const tankSpeed = TankInputTensor.speed[tankEid];
    const turretTarget = TankController.getTurretTarget(tankEid);
    const currentHealth = TankInputTensor.health[tankEid];
    const isShooting = actions[0] > 0.5;

    // Инициализируем пустую структуру наград
    const rewards = initializeRewards();

    // Рассчитываем отдельные компоненты наград
    const mapReward = calculateMapReward(tankX, tankY, width, height);
    const healthReward = calculateHealthReward(currentHealth, prevHealth);

    // Анализ целей и прицеливания
    const aimingResult = analyzeAiming(tankEid, tankX, tankY, turretTarget);
    const trackingReward = calculateTrackingReward(tankEid, turretTarget, aimingResult);
    const shootingReward = calculateShootingReward(isShooting, aimingResult.bestAimQuality);

    // Анализ избегания пуль
    const bulletAvoidanceResult = calculateBulletAvoidanceReward(tankEid);

    // Анализ движения
    const movementReward = calculateMovementReward(
        tankSpeed,
        bulletAvoidanceResult.maxDangerLevel,
        aimingResult.hasTargets,
        aimingResult.closestEnemyDist,
    );

    // Бонус за выживание
    const survivalReward = calculateSurvivalReward(currentHealth);

    // Заполняем структуру наград результатами расчетов
    rewards.common.health = healthReward.health;

    rewards.aim.accuracy = aimingResult.aimQualityReward;
    rewards.aim.tracking = trackingReward;

    rewards.shoot.aimQuality = aimingResult.aimQualityReward;
    rewards.shoot.shootDecision = shootingReward;

    rewards.movement.mapAwareness = mapReward;
    rewards.movement.speed = movementReward.speed;
    rewards.movement.avoidance = bulletAvoidanceResult.reward;
    rewards.movement.positioning = movementReward.positioning;

    rewards.common.survival = survivalReward;

    // Рассчитываем итоговые значения
    rewards.shoot.total = rewards.shoot.aimQuality + rewards.shoot.shootDecision;
    rewards.movement.total = rewards.movement.speed + rewards.movement.positioning +
        rewards.movement.avoidance + rewards.movement.mapAwareness;
    rewards.aim.total = rewards.aim.accuracy + rewards.aim.tracking;
    rewards.common.total = rewards.common.health + rewards.common.survival;

    // Общая итоговая награда
    rewards.totalReward = rewards.shoot.total + rewards.movement.total +
        rewards.aim.total + rewards.common.total;

    // Ограничиваем итоговое значение
    rewards.totalReward = Math.max(-10, Math.min(10, rewards.totalReward));

    isVerboseLog()
    && console.log(`Rewards common: ${ rewards.common.total.toFixed(2) }, aim: ${ rewards.aim.total.toFixed(2) }, shoot: ${ rewards.shoot.total.toFixed(2) }, movement: ${ rewards.movement.total.toFixed(2) }`);

    return rewards;
}

/**
 * Инициализация пустой структуры наград
 */
function initializeRewards(): ComponentRewards {
    return {
        shoot: { aimQuality: 0, shootDecision: 0, total: 0 },
        movement: { speed: 0, positioning: 0, avoidance: 0, mapAwareness: 0, total: 0 },
        aim: { accuracy: 0, tracking: 0, total: 0 },
        common: { health: 0, survival: 0, total: 0 },
        totalReward: 0,
    };
}

/**
 * Расчет награды за нахождение в пределах карты
 */
function calculateMapReward(
    tankX: number,
    tankY: number,
    width: number,
    height: number,
): number {
    if (tankX >= 0 && tankX <= width && tankY >= 0 && tankY <= height) {
        // Танк в пределах карты - используем плавный градиент
        const borderDistance = Math.min(
            tankX,
            tankY,
            width - tankX,
            height - tankY,
        );

        // Плавное уменьшение награды при приближении к границе
        const borderFactor = smoothstep(0, 150, borderDistance);
        return REWARD_WEIGHTS.BORDER_GRADIENT * (1 - borderFactor);
    } else {
        // Танк вышел за границы карты
        return REWARD_WEIGHTS.MAP_BORDER;
    }
}

/**
 * Расчет награды за сохранение здоровья
 */
function calculateHealthReward(
    currentHealth: number,
    prevHealth: number,
): { health: number } {
    // Награда за изменение здоровья
    const healthChange = currentHealth - prevHealth;
    const healthChangeReward = healthChange * REWARD_WEIGHTS.HEALTH_CHANGE;

    // Бонус за текущее здоровье
    const healthBonusReward = currentHealth * REWARD_WEIGHTS.HEALTH_BONUS;

    return {
        health: healthChangeReward + healthBonusReward,
    };
}

/**
 * Анализ прицеливания и видимых врагов
 */
function analyzeAiming(
    tankEid: number,
    tankX: number,
    tankY: number,
    turretTarget: Float64Array,
): {
    bestAimQuality: number;
    aimQualityReward: number;
    hasTargets: boolean;
    closestEnemyDist: number;
    enemyPositions: Map<number, [number, number]>;
} {
    let bestAimQuality = 0;
    let hasTargets = false;
    let closestEnemyDist = Number.MAX_VALUE;

    // Инициализация хранилища для позиций врагов
    if (!previousEnemyPositions.has(tankEid)) {
        previousEnemyPositions.set(tankEid, new Map());
    }

    // Текущие позиции врагов
    const currentEnemyPositions = new Map<number, [number, number]>();

    // Анализируем всех видимых врагов
    for (let j = 0; j < TANK_INPUT_TENSOR_MAX_ENEMIES; j++) {
        const enemyId = TankInputTensor.enemiesData.get(tankEid, j * 5);
        const enemyX = TankInputTensor.enemiesData.get(tankEid, j * 5 + 1);
        const enemyY = TankInputTensor.enemiesData.get(tankEid, j * 5 + 2);

        if (enemyId === 0) {
            continue;
        }

        hasTargets = true;
        const distToEnemy = hypot(tankX - enemyX, tankY - enemyY);

        // Сохраняем текущую позицию врага
        currentEnemyPositions.set(enemyId, [enemyX, enemyY]);

        // Обновляем дистанцию до ближайшего врага
        if (distToEnemy < closestEnemyDist) {
            closestEnemyDist = distToEnemy;
        }

        // Оценка точности прицеливания с плавным переходом
        const distFromTargetToEnemy = hypot(turretTarget[0] - enemyX, turretTarget[1] - enemyY);
        const targetRadius = TANK_RADIUS * 3;
        const aimQuality = Math.max(0, 1 - smoothstep(0, targetRadius, distFromTargetToEnemy));

        // Отслеживаем лучшее прицеливание
        if (aimQuality > bestAimQuality) {
            bestAimQuality = aimQuality;
        }
    }

    // Награда за качество прицеливания
    const aimQualityReward = bestAimQuality * REWARD_WEIGHTS.AIM_QUALITY;

    // Сохраняем текущие позиции врагов для следующего кадра
    previousEnemyPositions.set(tankEid, currentEnemyPositions);

    // Сохраняем текущую позицию прицела для следующего кадра
    previousTurretTargets.set(tankEid, [...turretTarget] as [number, number]);

    return {
        bestAimQuality,
        aimQualityReward,
        hasTargets,
        closestEnemyDist,
        enemyPositions: currentEnemyPositions,
    };
}

/**
 * Расчет награды за отслеживание целей
 */
function calculateTrackingReward(
    tankEid: number,
    turretTarget: Float64Array,
    aimingResult: ReturnType<typeof analyzeAiming>,
): number {
    let trackingReward = 0;

    // Проверяем движение прицела
    if (previousTurretTargets.has(tankEid)) {
        const prevTarget = previousTurretTargets.get(tankEid)!;

        // Расстояние, на которое переместился прицел
        const targetMovement = hypot(turretTarget[0] - prevTarget[0], turretTarget[1] - prevTarget[1]);

        // Штраф за статичный прицел (небольшое движение прицела допускается)
        if (targetMovement < 5) {
            trackingReward += REWARD_WEIGHTS.AIM_STATIC_PENALTY;
        }
    }

    // Проверяем отслеживание движущихся врагов
    if (previousEnemyPositions.has(tankEid) && previousTurretTargets.has(tankEid)) {
        const prevEnemyPositions = previousEnemyPositions.get(tankEid)!;
        const prevTurretTarget = previousTurretTargets.get(tankEid)!;

        // Перебираем всех текущих врагов
        for (const [enemyId, [enemyX, enemyY]] of aimingResult.enemyPositions.entries()) {
            // Если у нас есть информация о предыдущей позиции врага
            if (prevEnemyPositions.has(enemyId)) {
                const [prevEnemyX, prevEnemyY] = prevEnemyPositions.get(enemyId)!;

                // Расстояние, на которое переместился враг
                const enemyMovement = hypot(enemyX - prevEnemyX, enemyY - prevEnemyY);

                // Если враг двигался достаточно сильно
                if (enemyMovement > 5) {
                    // Вектор движения врага
                    const enemyDirX = enemyX - prevEnemyX;
                    const enemyDirY = enemyY - prevEnemyY;

                    // Вектор движения прицела
                    const turretDirX = turretTarget[0] - prevTurretTarget[0];
                    const turretDirY = turretTarget[1] - prevTurretTarget[1];

                    // Если прицел тоже двигался
                    if (Math.hypot(turretDirX, turretDirY) > 0.1) {
                        // Косинус угла между векторами (dot product / product of magnitudes)
                        const cosBetweenVectors =
                            (enemyDirX * turretDirX + enemyDirY * turretDirY) /
                            (Math.sqrt(enemyDirX * enemyDirX + enemyDirY * enemyDirY) *
                                Math.sqrt(turretDirX * turretDirX + turretDirY * turretDirY) || 1);

                        // Награда за отслеживание движения врага (максимум при cosBetweenVectors = 1)
                        if (cosBetweenVectors > 0) {
                            trackingReward += cosBetweenVectors * REWARD_WEIGHTS.AIM_TRACKING;
                        }
                    }
                }
            }
        }
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
        // Плавная награда за стрельбу в зависимости от точности прицеливания
        shootingReward += smoothstep(0.3, 0.8, bestAimQuality) * REWARD_WEIGHTS.SHOOTING_AIMED;

        // Штраф за стрельбу при плохом прицеливании (плавно уменьшается с ростом точности)
        if (bestAimQuality < 0.3) {
            shootingReward += smoothstep(0.3, 0, bestAimQuality) * REWARD_WEIGHTS.SHOOTING_RANDOM;
        }
    } else if (bestAimQuality > 0.8) {
        // Небольшой штраф за отсутствие стрельбы при хорошем прицеливании
        shootingReward -= REWARD_WEIGHTS.SHOOTING_AIMED * 0.3 * smoothstep(0.8, 1.0, bestAimQuality);
    }

    return shootingReward;
}

/**
 * Расчет награды за движение
 */
function calculateMovementReward(
    tankSpeed: number,
    maxDangerLevel: number,
    hasTargets: boolean,
    closestEnemyDist: number,
): { speed: number; positioning: number } {
    let speedReward = 0;
    let positioningReward = 0;

    // Базовая награда за движение (плавно растет с увеличением скорости)
    const speedFactor = smoothstep(0, 300, tankSpeed);
    speedReward += speedFactor * REWARD_WEIGHTS.MOVEMENT_BASE;

    // Стратегическое движение при наличии опасности
    if (maxDangerLevel > 0.3) {
        // Дополнительная награда за движение при наличии опасных пуль
        const dangerSpeedBonus = speedFactor * REWARD_WEIGHTS.STRATEGIC_MOVEMENT *
            smoothstep(0.3, 1.0, maxDangerLevel);
        speedReward += dangerSpeedBonus;
    }

    // Награда за позиционирование относительно врагов
    if (hasTargets) {
        // Оптимальная дистанция: от 3*TANK_RADIUS до 500
        if (closestEnemyDist < 3 * TANK_RADIUS) {
            // Штраф за слишком близкое расстояние
            const tooClosePenalty = smoothstep(TANK_RADIUS, 3 * TANK_RADIUS, closestEnemyDist) - 1;
            positioningReward += tooClosePenalty * 0.3;
        } else if (closestEnemyDist <= 500) {
            // Награда за оптимальную дистанцию
            const optimalDistanceReward = smoothstep(3 * TANK_RADIUS, 200, closestEnemyDist) *
                (1 - smoothstep(200, 500, closestEnemyDist));
            positioningReward += optimalDistanceReward * REWARD_WEIGHTS.DISTANCE_KEEPING;
        } else {
            // Мягкий штраф за слишком большую дистанцию
            const tooFarPenalty = smoothstep(500, 800, closestEnemyDist) * -0.1;
            positioningReward += tooFarPenalty;
        }
    }

    return { speed: speedReward, positioning: positioningReward };
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
 * Расчет награды за избегание пуль с использованием плавных переходов
 */
function calculateBulletAvoidanceReward(tankEid: number) {
    let reward = 0;
    const tankX = TankInputTensor.x[tankEid];
    const tankY = TankInputTensor.y[tankEid];

    // Инициализируем хранилище пуль для танка, если его еще нет
    if (!bulletMemory.has(tankEid)) {
        bulletMemory.set(tankEid, new Map());
    }

    // Получаем хранилище пуль для танка
    const tankBulletMemory = bulletMemory.get(tankEid)!;

    // Счетчики для анализа
    let avoidedBullets = 0;
    let dangerousBullets = 0;
    let maxDangerLevel = 0;

    // Получаем данные о пулях из тензора
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_BULLETS; i++) {
        const id = TankInputTensor.bulletsData.get(tankEid, i * 5);
        const x = TankInputTensor.bulletsData.get(tankEid, i * 5 + 1);
        const y = TankInputTensor.bulletsData.get(tankEid, i * 5 + 2);
        const vx = TankInputTensor.bulletsData.get(tankEid, i * 5 + 3);
        const vy = TankInputTensor.bulletsData.get(tankEid, i * 5 + 4);

        if (id === 0) {
            continue;
        }

        // Анализируем пулю
        const bulletResult = analyzeBullet(
            tankX, tankY, id, x, y, vx, vy, tankBulletMemory,
        );

        // Обновляем статистику
        reward += bulletResult.reward;

        if (bulletResult.isDangerous) {
            dangerousBullets++;
            maxDangerLevel = Math.max(maxDangerLevel, bulletResult.dangerLevel);
        }

        if (bulletResult.wasAvoided) {
            avoidedBullets++;
        }
    }

    // Очистка устаревших записей о пулях (старше 5 секунд)
    const now = Date.now();
    for (const [bulletId, bulletInfo] of tankBulletMemory.entries()) {
        if (now - bulletInfo.timestamp > 5000) {
            tankBulletMemory.delete(bulletId);
        }
    }

    // Дополнительный бонус за отсутствие опасных пуль при наличии избежанных
    if (dangerousBullets === 0 && avoidedBullets > 0) {
        // Бонус растет нелинейно с количеством избежанных пуль
        reward += Math.min(1.5, avoidedBullets * 0.2);
    }

    return { reward, maxDangerLevel, avoidedBullets, dangerousBullets };
}


/**
 * Анализ опасности пули для танка
 */
function analyzeBullet(
    tankX: number,
    tankY: number,
    bulletId: number,
    bulletX: number,
    bulletY: number,
    bulletVx: number,
    bulletVy: number,
    tankBulletMemory: Map<number, BulletInfo>,
): { reward: number; isDangerous: boolean; wasAvoided: boolean; dangerLevel: number } {
    // Проверяем, новая ли это пуля
    if (!tankBulletMemory.has(bulletId)) {
        // Добавляем новую пулю в память
        tankBulletMemory.set(bulletId, {
            id: bulletId,
            x: bulletX,
            y: bulletY,
            vx: bulletVx,
            vy: bulletVy,
            timestamp: Date.now(),
            wasAvoided: false,
        });
    }

    // Получаем информацию о пуле
    const bulletInfo = tankBulletMemory.get(bulletId)!;

    // Результаты анализа
    let reward = 0;
    let isDangerous = false;
    let wasAvoided = false;
    let dangerLevel = 0;

    // Анализируем опасность пули
    const bulletSpeed = Math.hypot(bulletVx, bulletVy);
    if (bulletSpeed < 0.001) return { reward, isDangerous, wasAvoided, dangerLevel };

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
        const minDist = Math.hypot(closestPointX - tankX, closestPointY - tankY);

        // Оценка опасности пули с плавным переходом
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

                // Награда пропорциональна уровню опасности
                reward += dangerLevel * REWARD_WEIGHTS.BULLET_AVOIDANCE;
            }
        } else {
            // Пуля пролетела далеко от танка
            if (!bulletInfo.wasAvoided) {
                bulletInfo.wasAvoided = true;
                wasAvoided = true;

                // Плавное увеличение награды в зависимости от того, насколько близко была пуля
                const avoidFactor = smoothstep(200, 120, minDist);
                reward += 0.5 * (1 + avoidFactor); // От 0.5 до 1.0
            }
        }
    }

    return { reward, isDangerous, wasAvoided, dangerLevel };
}
