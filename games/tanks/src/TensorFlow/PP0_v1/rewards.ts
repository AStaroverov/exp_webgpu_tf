// Enhanced reward calculation with simplified structure
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';
import { TankController } from '../../ECS/Components/TankController.ts';
import { hypot } from '../../../../../lib/math.ts';
import { TANK_RADIUS } from './consts.ts';

// Константы для калибровки вознаграждений
const REWARD_WEIGHTS = {
    HEALTH_CHANGE: 7.0,        // За потерю всего здоровья (от 1 до 0) штраф -7
    HEALTH_BONUS: 0.05,        // За полное здоровье бонус +0.05 (незначительный)
    AIM_QUALITY: 2.5,                  // Максимум +2.5 за идеальное прицеливание
    SHOOTING_AIMED: 1.0,        // +1.0 за стрельбу при хорошем прицеливании
    SHOOTING_RANDOM: -0.5,      // -0.5 за случайную стрельбу
    BULLET_AVOIDANCE: -2.0,     // До -2.0 за нахождение на пути пули
    MOVEMENT_BASE: 0.05,        // Незначительный бонус за движение
    STRATEGIC_MOVEMENT: 0.2,    // Небольшой бонус за стратегическое движение
    SURVIVAL: 0.02,             // Маленький постоянный бонус за выживание
    MAP_BORDER: -3.0,           // Значительный штраф за выход за границы
    BORDER_GRADIENT: -0.5,      // Умеренный штраф при приближении к границе
    DISTANCE_KEEPING: 0.3,      // Бонус за поддержание оптимальной дистанции
    VICTORY: 10.0,              // Полный бонус +10 за победу (лучший исход)
    DEATH: -5.0,                 // Существенный штраф -5 за уничтожение (но не самый худший)
};

// Хранение предыдущих состояний для каждого танка
const prevTankStates = new Map<number, {
    // Последние N позиций танка для определения застревания
    positions: Array<{ x: number, y: number }>,
    // Количество последовательных тиков для определения однотипного поведения
    sameActionCounter: number,
    // Последнее действие
    lastAction?: number[]
}>();

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

    // Инициализируем состояние танка, если оно еще не существует
    if (!prevTankStates.has(tankEid)) {
        prevTankStates.set(tankEid, {
            positions: Array(10).fill({ x: tankX, y: tankY }),
            sameActionCounter: 0,
        });
    }

    // Получаем предыдущее состояние
    const tankState = prevTankStates.get(tankEid)!;

    // Обновляем историю позиций (для определения застревания)
    tankState.positions.push({ x: tankX, y: tankY });
    if (tankState.positions.length > 10) {
        tankState.positions.shift();
    }

    // Проверка на повторяющиеся действия
    let repeatedActionPenalty = 0;
    if (tankState.lastAction) {
        // Сравниваем текущее и предыдущее действия
        let actionSimilarity = 0;
        for (let i = 0; i < 5; i++) {
            if (Math.abs((actions[i] || 0) - (tankState.lastAction[i] || 0)) < 0.1) {
                actionSimilarity++;
            }
        }

        // Если все 5 действий почти идентичны
        if (actionSimilarity >= 5) {
            tankState.sameActionCounter++;
            // Применяем штраф только если действия повторяются долго
            if (tankState.sameActionCounter > 20) {
                repeatedActionPenalty = -0.05 * (tankState.sameActionCounter - 20);
                // Ограничиваем максимальный штраф
                repeatedActionPenalty = Math.max(repeatedActionPenalty, -0.5);
            }
        } else {
            tankState.sameActionCounter = 0;
        }
    }

    // Сохраняем текущее действие для следующего сравнения
    tankState.lastAction = Array.from(actions);

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
        exploration: 0,      // Исследование карты
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
        const enemyX = TankInputTensor.enemiesData.get(tankEid, j * 4);
        const enemyY = TankInputTensor.enemiesData.get(tankEid, j * 4 + 1);

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
    let bulletDangerLevel = 0;
    let closestBulletDistance = Number.MAX_VALUE;

    for (let j = 0; j < TANK_INPUT_TENSOR_MAX_BULLETS; j++) {
        const bulletX = TankInputTensor.bulletsData.get(tankEid, j * 4);
        const bulletY = TankInputTensor.bulletsData.get(tankEid, j * 4 + 1);
        const bulletVx = TankInputTensor.bulletsData.get(tankEid, j * 4 + 2);
        const bulletVy = TankInputTensor.bulletsData.get(tankEid, j * 4 + 3);

        if ((bulletX === 0 && bulletY === 0) || hypot(bulletVx, bulletVy) < 100) continue;

        // Вектор от пули к танку
        const toTankX = tankX - bulletX;
        const toTankY = tankY - bulletY;

        // Скорость пули и ее направление
        const bulletSpeed = hypot(bulletVx, bulletVy);
        const bulletDirectionX = bulletSpeed > 0.001 ? bulletVx / bulletSpeed : 0;
        const bulletDirectionY = bulletSpeed > 0.001 ? bulletVy / bulletSpeed : 0;

        // Определяем, движется ли пуля к танку
        const dotProduct = toTankX * bulletDirectionX + toTankY * bulletDirectionY;

        // Если пуля движется от танка, игнорируем
        if (dotProduct <= 0) continue;

        // Расстояние от танка до пули
        const bulletDist = hypot(toTankX, toTankY);
        closestBulletDistance = Math.min(closestBulletDistance, bulletDist);

        // Проекция вектора на направление пули
        const projLength = dotProduct;

        // Точка ближайшего прохождения пули к танку
        const closestPointX = bulletX + bulletDirectionX * projLength;
        const closestPointY = bulletY + bulletDirectionY * projLength;

        // Расстояние в точке наибольшего сближения
        const minDist = hypot(closestPointX - tankX, closestPointY - tankY);

        // Оценка опасности пули
        if (minDist < TANK_RADIUS * 2) {
            // Время до точки сближения
            const timeToClosest = projLength / bulletSpeed;

            // Чем ближе пуля и меньше времени, тем выше опасность
            if (timeToClosest < 1.0) {
                bulletDangerLevel += (1.0 - minDist / (TANK_RADIUS * 2)) * (1.0 - timeToClosest);
            }
        }
    }

    // Награда за избегание пуль - упрощена
    if (bulletDangerLevel > 0) {
        rewardRecord.avoidBullets = bulletDangerLevel * REWARD_WEIGHTS.BULLET_AVOIDANCE;

        // Если танк движется при наличии опасных пуль - уменьшаем штраф
        if (tankSpeed > 200) {
            rewardRecord.avoidBullets *= 0.5; // Уменьшаем штраф вдвое если танк активно движется
        }
    }

    // 5. Награда за движение - упрощено
    rewardRecord.movement = 0;

    // Базовый бонус за движение
    if (tankSpeed > 50) {
        rewardRecord.movement += REWARD_WEIGHTS.MOVEMENT_BASE;
    }

    // Стратегическое движение в зависимости от ситуации
    if (bulletDangerLevel > 0.3 && tankSpeed > 200) {
        // Бонус за быстрое движение при опасности
        rewardRecord.movement += REWARD_WEIGHTS.STRATEGIC_MOVEMENT;
    } else if (closestEnemyDist < 150 && tankSpeed > 100) {
        // Бонус за движение при близком враге
        rewardRecord.movement += REWARD_WEIGHTS.STRATEGIC_MOVEMENT;
    } else if (!hasTargets && tankSpeed > 100) {
        // Бонус за исследование когда нет видимых врагов
        rewardRecord.movement += REWARD_WEIGHTS.STRATEGIC_MOVEMENT;
    }

    // Проверка на застревание (перемещение меньше порога)
    if (tankState.positions.length >= 10) {
        let totalMovement = 0;
        for (let i = 1; i < tankState.positions.length; i++) {
            totalMovement += hypot(
                tankState.positions[i].x - tankState.positions[i - 1].x,
                tankState.positions[i].y - tankState.positions[i - 1].y,
            );
        }

        // Если танк практически не двигался за последние 10 тиков
        if (totalMovement < 50) {
            rewardRecord.movement -= 0.2; // Штраф за застревание
        }
    }

    // 6. Поддержание оптимальной дистанции от врагов
    if (hasTargets) {
        if (closestEnemyDist < 150) {
            // Слишком близко - опасно
            rewardRecord.avoidEnemies = -0.2;
        } else if (closestEnemyDist > 200 && closestEnemyDist < 500) {
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

    // Финальный штраф за повторяющиеся действия
    if (repeatedActionPenalty !== 0) {
        rewardRecord.exploration = repeatedActionPenalty;
    }

    const totalReward =
        rewardRecord.map +
        rewardRecord.health +
        rewardRecord.aim +
        rewardRecord.avoidBullets +
        rewardRecord.avoidEnemies +
        rewardRecord.movement +
        rewardRecord.survival +
        rewardRecord.exploration;

    const clippedReward = Math.max(-15, Math.min(15, totalReward));

    if (Math.abs(totalReward - clippedReward) > 3.0) {
        console.warn('Warning: Extreme reward clipped from', totalReward, 'to', clippedReward);
    }

    return {
        totalReward: clippedReward,
        rewards: rewardRecord,
    };
}