// Enhanced reward calculation
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../ECS/Components/TankState.ts';
import { TankController } from '../ECS/Components/TankController.ts';
import { inRange } from 'lodash-es';
import { hypot, max, smoothstep } from '../../../../lib/math.ts';
import { TANK_RADIUS } from './consts.ts';

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

    // Расширенные компоненты вознаграждения
    const rewardRecord = {
        map: 0,              // Нахождение в пределах карты
        aim: 0,              // Наведение на врагов
        avoidBullets: 0,     // Избегание пуль
        avoidEnemies: 0,     // Поддержание дистанции от врагов
        health: 0,           // Сохранение/потеря здоровья
        damageDealt: 0,      // Нанесение урона -- TODO: Добавить
        movement: 0,         // Эффективность движения
        survival: 0,         // Бонус за выживание
    };

    // 1. Reward for staying within map (with distance-based gradient)
    if (inRange(tankX, 0, width) && inRange(tankY, 0, height)) {
        // Базовое вознаграждение за нахождение в пределах карты
        rewardRecord.map = 1.0;

        // Дополнительный штраф, если танк близок к границе
        const distToBorder = Math.min(
            tankX,
            tankY,
            width - tankX,
            height - tankY,
        );

        // Если ближе 50 единиц к границе - начинаем уменьшать награду
        if (distToBorder < 50) {
            rewardRecord.map -= (1 - distToBorder / 50) * 0.8;
        }
    } else {
        // Существенный штраф за выход за пределы карты
        rewardRecord.map = -2.0;
    }

    // 2. Reward/penalty for health changes
    const healthChange = currentHealth - prevHealth;
    if (healthChange < 0) {
        // Штраф за потерю здоровья
        rewardRecord.health = healthChange * 0.5; // Умножаем на 0.5 для смягчения штрафа
    } else if (healthChange > 0) {
        // Бонус за восстановление здоровья (если такая механика присутствует)
        rewardRecord.health = healthChange * 0.3;
    }

    // Базовое вознаграждение за оставшееся здоровье
    rewardRecord.health += currentHealth * 0.05;

    // 3. Reward for aiming at enemies
    let hasTargets = false;
    let closestEnemyDist = Number.MAX_VALUE;
    let enemiesNearby = 0;

    for (let j = 0; j < TANK_INPUT_TENSOR_MAX_ENEMIES; j++) {
        const enemyX = TankInputTensor.enemiesData.get(tankEid, j * 4);
        const enemyY = TankInputTensor.enemiesData.get(tankEid, j * 4 + 1);
        const enemyVx = TankInputTensor.enemiesData.get(tankEid, j * 4 + 2);
        const enemyVy = TankInputTensor.enemiesData.get(tankEid, j * 4 + 3);

        // Проверяем, что враг существует
        if (enemyX !== 0 || enemyY !== 0) {
            hasTargets = true;
            const distFromTankToEnemy = hypot(tankX - enemyX, tankY - enemyY);

            // Обновляем дистанцию до ближайшего врага
            if (distFromTankToEnemy < closestEnemyDist) {
                closestEnemyDist = distFromTankToEnemy;
            }

            // Подсчитываем врагов в радиусе 400 единиц
            if (distFromTankToEnemy < 400) {
                enemiesNearby++;
            }

            // Вознаграждение за наведение
            const distFromTargetToEnemy = hypot(turretTarget[0] - enemyX, turretTarget[1] - enemyY);

            // Прогнозируем будущую позицию врага (упрощенно)
            const futureEnemyX = enemyX + enemyVx * 0.5; // Прогноз на 0.5 секунд
            const futureEnemyY = enemyY + enemyVy * 0.5;
            const distToFuturePosition = hypot(turretTarget[0] - futureEnemyX, turretTarget[1] - futureEnemyY);

            // Учитываем как текущую, так и прогнозируемую позицию
            const aimQuality = Math.max(
                1 - distFromTargetToEnemy / TANK_RADIUS,
                1 - distToFuturePosition / TANK_RADIUS,
            );

            // Масштабируем по расстоянию (ближе = важнее)
            const distanceImportance = 1 - smoothstep(200, 800, distFromTankToEnemy);

            rewardRecord.aim += max(0, aimQuality) * distanceImportance;

            // Вознаграждение за поддержание безопасной дистанции
            if (distFromTankToEnemy < 150) {
                // Слишком близко - опасно
                rewardRecord.avoidEnemies -= 0.2;
            } else if (distFromTankToEnemy > 200 && distFromTankToEnemy < 600) {
                // Идеальный диапазон
                rewardRecord.avoidEnemies += 0.15;
            } else if (distFromTankToEnemy > 800) {
                // Слишком далеко - малоэффективно
                rewardRecord.avoidEnemies -= 0.05;
            }
        }
    }

    // Если цели есть, и прицелился хорошо, и стреляет - дополнительный бонус
    const shouldShoot = actions && actions[0] > 0;
    if (hasTargets && rewardRecord.aim > 0.7 && shouldShoot) {
        rewardRecord.aim += 0.5;
    } else if (shouldShoot && rewardRecord.aim < 0.3) {
        // Штраф за стрельбу без прицеливания
        rewardRecord.aim -= 0.2;
    }

    // 4. Reward for avoiding bullets
    let bulletDangerLevel = 0;

    for (let j = 0; j < TANK_INPUT_TENSOR_MAX_BULLETS; j++) {
        const bulletX = TankInputTensor.bulletsData.get(tankEid, j * 4);
        const bulletY = TankInputTensor.bulletsData.get(tankEid, j * 4 + 1);
        const bulletVx = TankInputTensor.bulletsData.get(tankEid, j * 4 + 2);
        const bulletVy = TankInputTensor.bulletsData.get(tankEid, j * 4 + 3);

        if ((bulletX === 0 && bulletY === 0) || hypot(bulletVx, bulletVy) < 100) continue;

        // Вычисляем точку ближайшего прохождения пули
        // (Упрощенно, используем линейное приближение траектории)
        const bulletSpeed = hypot(bulletVx, bulletVy);
        const bulletDirectionX = bulletSpeed > 0.001 ? bulletVx / bulletSpeed : 0;
        const bulletDirectionY = bulletSpeed > 0.001 ? bulletVy / bulletSpeed : 0;
        // Вектор от пули к танку
        const toTankX = tankX - bulletX;
        const toTankY = tankY - bulletY;

        // Скалярное произведение для определения, движется ли пуля к танку
        const dotProduct = toTankX * bulletDirectionX + toTankY * bulletDirectionY;

        // Если пуля движется от танка, игнорируем ее
        if (dotProduct <= 0) continue;

        // Проекция вектора toTank на направление пули
        const projLength = dotProduct;

        // Точка ближайшего прохождения
        const closestPointX = bulletX + bulletDirectionX * projLength;
        const closestPointY = bulletY + bulletDirectionY * projLength;

        // Минимальное расстояние до траектории пули
        const minDist = hypot(closestPointX - tankX, closestPointY - tankY);

        // Если пуля пройдет близко к танку, увеличиваем опасность
        if (minDist < TANK_RADIUS * 1.5) {
            // Время до точки ближайшего прохождения
            const timeToClosest = projLength / bulletSpeed;

            // Чем меньше времени, тем выше опасность
            if (timeToClosest < 1.0) {
                // Очень высокая опасность для близких пуль
                bulletDangerLevel += (1.0 - minDist / (TANK_RADIUS * 1.5)) * (1.0 - timeToClosest);
            }
        }
    }

    // Штраф за нахождение на траектории пуль
    if (bulletDangerLevel > 0) {
        rewardRecord.avoidBullets = -bulletDangerLevel * 0.4;

        // Если танк движется (пытается уклониться) - снижаем штраф
        if (tankSpeed > 300) {
            rewardRecord.avoidBullets *= 0.7;
        }
    } else {
        // Небольшое вознаграждение за отсутствие опасных пуль рядом
        rewardRecord.avoidBullets = 0.05;
    }

    // 5. Reward for effective movement
    // Анализируем движение в зависимости от ситуации
    if (enemiesNearby > 1) {
        // При нескольких врагах рядом, движение важно для выживания
        if (tankSpeed > 200) {
            rewardRecord.movement = 0.2;
        }
    } else if (closestEnemyDist < 200) {
        // При близком враге, движение должно быть для уклонения
        if (tankSpeed > 200) {
            rewardRecord.movement = 0.15;
        }
    } else if (bulletDangerLevel > 0.3) {
        // При опасных пулях рядом, поощряем движение
        if (tankSpeed > 300) {
            rewardRecord.movement = 0.25;
        }
    } else if (!hasTargets) {
        // Если враги не видны, поощряем исследование
        if (tankSpeed > 100) {
            rewardRecord.movement = 0.1;
        }
    } else {
        // В обычных условиях, небольшое поощрение за умеренное движение
        if (tankSpeed > 50 && tankSpeed < 500) {
            rewardRecord.movement = 0.05;
        }
    }

    // 6. Survival bonus - награда за каждый шаг выживания
    rewardRecord.survival = 0.01;

    // Дополнительный бонус за выживание с низким здоровьем
    if (currentHealth < 0.3) {
        rewardRecord.survival += 0.02;
    }

    // Рассчитываем общее вознаграждение с разными весами для каждого компонента
    const totalReward =
        rewardRecord.map * 3.0 +
        rewardRecord.aim * 10.0 +
        rewardRecord.avoidBullets * 5.0 +
        rewardRecord.avoidEnemies * 3.0 +
        rewardRecord.health * 2.0 +
        rewardRecord.damageDealt * 8.0 +
        rewardRecord.movement * 2.0 +
        rewardRecord.survival * 1.0;

    if (totalReward < -20 || totalReward > 20) {
        console.log('>> unexpected reward:', totalReward, rewardRecord);
    }

    return {
        totalReward,
        rewards: rewardRecord,
    };
}