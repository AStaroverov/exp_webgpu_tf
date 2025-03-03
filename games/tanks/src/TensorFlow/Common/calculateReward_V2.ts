// Enhanced reward calculation with simplified structure
import { TANK_INPUT_TENSOR_MAX_ENEMIES, TankInputTensor } from '../../ECS/Components/TankState.ts';
import { TankController } from '../../ECS/Components/TankController.ts';
import { hypot } from '../../../../../lib/math.ts';
import { TANK_RADIUS } from './consts.ts';

// Константы для калибровки вознаграждений
const REWARD_WEIGHTS = {
    HEALTH_CHANGE: 0.5,
    HEALTH_BONUS: 0.05,
    AIM_QUALITY: 1,
    SHOOTING_AIMED: 0.5,
    SHOOTING_RANDOM: -0.5,
    BULLET_AVOIDANCE: -4.0,
    MOVEMENT_BASE: 0.2,
    STRATEGIC_MOVEMENT: 0.6,
    SURVIVAL: 0.02,
    MAP_BORDER: -3.0,
    BORDER_GRADIENT: -0.5,
    DISTANCE_KEEPING: 1,
    VICTORY: 10.0,
    DEATH: -5.0,
};

export function calculateReward(
    tankEid: number,
    actions: ArrayLike<number>,
    prevHealth: number,
    width: number,
    height: number,
): { totalReward: number, rewards: Record<string, number> } {
    const tankX = TankInputTensor.x[tankEid];
    const tankY = TankInputTensor.y[tankEid];
    const currentHealth = TankInputTensor.health[tankEid];
    const tankSpeed = TankInputTensor.speed[tankEid];
    const turretTarget = TankController.getTurretTarget(tankEid);

    // Компоненты вознаграждения
    const rewardRecord = {
        map: 0,              // Нахождение в пределах карты
        aim: 0,              // Наведение на врагов
        avoidBullets: 0,     // Избегание пуль
        avoidEnemies: 0,     // Поддержание дистанции от врагов
        health: 0,           // Сохранение/потеря здоровья
        damageDealt: 0,      // Нанесение урона
        movement: 0,         // Эффективность движения
        survival: 0,         // Бонус за выживание
    };

    // 1. Базовое вознаграждение за пребывание в пределах карты - упрощено
    if (tankX >= 0 && tankX <= width && tankY >= 0 && tankY <= height) {
        rewardRecord.map = 0; // Нейтральное вознаграждение в пределах карты

        // Градиентный штраф при приближении к краю
        const borderDistance = Math.min(
            tankX,
            tankY,
            width - tankX,
            height - tankY,
        );

        if (borderDistance < 50) {
            // Линейный штраф: от 0 до BORDER_GRADIENT
            rewardRecord.map = (borderDistance / 50) * REWARD_WEIGHTS.BORDER_GRADIENT;
        }
    } else {
        // Существенный штраф за выход за границы
        rewardRecord.map = REWARD_WEIGHTS.MAP_BORDER;
    }

    // 2. Здоровье - сильно упрощено и с высоким весом
    const healthChange = currentHealth - prevHealth;
    rewardRecord.health = healthChange * REWARD_WEIGHTS.HEALTH_CHANGE;

    // Небольшой бонус за оставшееся здоровье (мотивация выживать)
    rewardRecord.health += currentHealth * REWARD_WEIGHTS.HEALTH_BONUS;

    // 3. Прицеливание - упрощенная логика
    let bestAimQuality = 0;
    let hasTargets = false;
    let closestEnemyDist = Number.MAX_VALUE;

    for (let j = 0; j < TANK_INPUT_TENSOR_MAX_ENEMIES; j++) {
        const enemyX = TankInputTensor.enemiesData.get(tankEid, j * 5 + 1);
        const enemyY = TankInputTensor.enemiesData.get(tankEid, j * 5 + 2);

        // Если враг существует (не нулевые координаты)
        if (enemyX !== 0 || enemyY !== 0) {
            hasTargets = true;
            const distToEnemy = hypot(tankX - enemyX, tankY - enemyY);

            // Обновляем дистанцию до ближайшего врага
            closestEnemyDist = Math.min(closestEnemyDist, distToEnemy);

            // Простая оценка точности прицеливания
            const distFromTargetToEnemy = hypot(turretTarget[0] - enemyX, turretTarget[1] - enemyY);
            const aimQuality = Math.max(0, 1 - distFromTargetToEnemy / (TANK_RADIUS * 3));

            // Сохраняем лучшее прицеливание
            bestAimQuality = Math.max(bestAimQuality, aimQuality);
        }
    }

    // Устанавливаем награду за прицеливание
    rewardRecord.aim = bestAimQuality * REWARD_WEIGHTS.AIM_QUALITY;

    // Бонус/штраф за стрельбу в зависимости от точности прицеливания
    const shouldShoot = actions[0] > 0;
    if (shouldShoot) {
        if (bestAimQuality > 0.7) {
            // Бонус за хорошее прицеливание при стрельбе
            rewardRecord.aim += REWARD_WEIGHTS.SHOOTING_AIMED;
        } else if (bestAimQuality < 0.3) {
            // Штраф за стрельбу без прицеливания
            rewardRecord.aim += REWARD_WEIGHTS.SHOOTING_RANDOM;
        }
    }

    // 4. Уклонение от пуль - упрощено
    const { reward, maxDangerLevel } = calculateBulletAvoidanceReward(tankEid);
    rewardRecord.avoidBullets = reward;

    // 5. Награда за движение - упрощено
    rewardRecord.movement = 0;

    // Базовый бонус за движение
    if (tankSpeed > 50) {
        rewardRecord.movement += REWARD_WEIGHTS.MOVEMENT_BASE;
    }

    // Стратегическое движение в зависимости от ситуации
    if (maxDangerLevel > 0.3 && tankSpeed > 200) {
        // Бонус за быстрое движение при опасности
        rewardRecord.movement += REWARD_WEIGHTS.STRATEGIC_MOVEMENT;
    } else if (closestEnemyDist < TANK_RADIUS * 3 && tankSpeed > 100) {
        // Бонус за движение при близком враге
        rewardRecord.movement += REWARD_WEIGHTS.STRATEGIC_MOVEMENT;
    } else if (!hasTargets && tankSpeed > 100) {
        // Бонус за исследование когда нет видимых врагов
        rewardRecord.movement += REWARD_WEIGHTS.STRATEGIC_MOVEMENT;
    }

    // 6. Поддержание оптимальной дистанции от врагов
    if (hasTargets) {
        if (closestEnemyDist < TANK_RADIUS * 3) {
            // Слишком близко - опасно
            rewardRecord.avoidEnemies = -0.2;
        } else if (closestEnemyDist > TANK_RADIUS * 3 && closestEnemyDist < 600) {
            // Оптимальная дистанция
            rewardRecord.avoidEnemies = REWARD_WEIGHTS.DISTANCE_KEEPING;
        }
    }

    // 7. Бонус за выживание
    rewardRecord.survival = REWARD_WEIGHTS.SURVIVAL;

    // Дополнительный бонус при низком здоровье
    if (currentHealth < 0.3) {
        rewardRecord.survival += REWARD_WEIGHTS.SURVIVAL;
    }

    const totalReward =
        rewardRecord.map +
        rewardRecord.health +
        rewardRecord.aim +
        rewardRecord.avoidBullets +
        rewardRecord.avoidEnemies +
        rewardRecord.movement +
        rewardRecord.survival;

    const clippedReward = Math.max(-15, Math.min(15, totalReward));

    if (Math.abs(totalReward - clippedReward) > 3.0) {
        console.warn('Warning: Extreme reward clipped from', totalReward, 'to', clippedReward);
    }

    return {
        totalReward: clippedReward,
        rewards: rewardRecord,
    };
}

export type BulletInfo = {
    id: number;
    x: number;
    y: number;
    vx: number;
    vy: number;
    timestamp: number;
    wasAvoided: boolean;
};

const bulletMemory: Map<number, BulletInfo> = new Map();

function calculateBulletAvoidanceReward(tankEid: number) {
    let reward = 0;
    const tankX = TankInputTensor.x[tankEid];
    const tankY = TankInputTensor.y[tankEid];

    // Подсчет количества успешно избежанных пуль
    let avoidedBullets = 0;
    let dangerousBullets = 0;
    let maxDangerLevel = 0;

    for (const bulletInfo of bulletMemory.values()) {
        // Для избежанных пуль
        if (bulletInfo.wasAvoided) {
            avoidedBullets++;
            reward += 0.5; // Бонус за каждую избежанную пулю
        } else {
            // Для текущих опасных пуль
            const bulletSpeed = Math.hypot(bulletInfo.vx, bulletInfo.vy);
            const bulletDirectionX = bulletInfo.vx / bulletSpeed;
            const bulletDirectionY = bulletInfo.vy / bulletSpeed;

            // Вектор от пули к танку
            const toTankX = tankX - bulletInfo.x;
            const toTankY = tankY - bulletInfo.y;

            // Определяем, движется ли пуля к танку
            const dotProduct = toTankX * bulletDirectionX + toTankY * bulletDirectionY;

            // Если пуля движется к танку
            if (dotProduct > 0) {
                // Проекция вектора на направление пули
                const projLength = dotProduct;

                // Точка ближайшего прохождения пули к танку
                const closestPointX = bulletInfo.x + bulletDirectionX * projLength;
                const closestPointY = bulletInfo.y + bulletDirectionY * projLength;

                // Расстояние в точке наибольшего сближения
                const minDist = Math.hypot(closestPointX - tankX, closestPointY - tankY);

                // Оценка опасности пули
                if (minDist < 100) { // приблизительный двойной радиус танка
                    // Время до точки сближения
                    const timeToClosest = projLength / bulletSpeed;

                    // Чем ближе пуля и меньше времени, тем выше опасность
                    if (timeToClosest < 1.0) {
                        const dangerLevel = (1.0 - minDist / 100) * (1.0 - timeToClosest);
                        maxDangerLevel = Math.max(maxDangerLevel, dangerLevel);
                        dangerousBullets++;
                        reward -= dangerLevel * REWARD_WEIGHTS.BULLET_AVOIDANCE;
                    }
                }
            }
        }
    }

    // Дополнительный бонус за отсутствие опасных пуль при наличии избежанных
    if (dangerousBullets === 0 && avoidedBullets > 0) {
        reward += avoidedBullets * 0.2;
    }

    return { reward, maxDangerLevel };
}