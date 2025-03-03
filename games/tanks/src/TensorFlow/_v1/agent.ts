import * as tf from '@tensorflow/tfjs';
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState';
import { TankController } from '../../ECS/Components/TankController';

import { createInputVector } from '../Common/createInputVector.ts';
import { REWARD_WEIGHTS } from './const.ts';
import { addExperience, ModelPrediction, predictActions, ReplayBuffer } from './model.ts';

// Типы для агента танка
export type TankState = {
    tankEid: number;
    lastState: Float32Array | null;
    lastAction: { movement: number[], aim: number[], shoot: boolean } | null;
    lastHealth: number;
    episodeRewards: number[];
    rewardHistory: { [key: string]: number[] };
    bulletMemory: Map<string, BulletInfo>;
};

export type BulletInfo = {
    id: number;
    x: number;
    y: number;
    vx: number;
    vy: number;
    timestamp: number;
    wasAvoided: boolean;
};

export type RewardComponents = {
    map: number;
    aim: number;
    avoidBullets: number;
    avoidEnemies: number;
    health: number;
    damageDealt: number;
    movement: number;
    survival: number;
};

export type RewardResult = {
    totalReward: number;
    rewards: RewardComponents;
};

export type AgentStats = {
    episodeRewards: number[];
    rewardHistory: { [key: string]: number[] };
    bulletMemory: {
        count: number;
        avoided: number;
    };
};

// Создание начального состояния агента
export function createAgentState(tankEid: number): TankState {
    return {
        tankEid,
        lastState: null,
        lastAction: null,
        lastHealth: 1.0,
        episodeRewards: [],
        rewardHistory: {
            map: [],
            aim: [],
            avoidBullets: [],
            avoidEnemies: [],
            health: [],
            damageDealt: [],
            movement: [],
            survival: [],
        },
        bulletMemory: new Map(),
    };
}

// Выбор действия и его исполнение
export function actAgent(
    agentState: TankState,
    model: tf.LayersModel,
    replayBuffer: ReplayBuffer,
    width: number,
    height: number,
    maxSpeed: number,
): ModelPrediction {
    // Получение текущего вектора состояния
    const currentState = createInputVector(agentState.tankEid, width, height, maxSpeed);

    // Получение предсказаний модели
    const action = predictActions(model, agentState.tankEid, width, height, maxSpeed);

    // Сохранение текущего состояния и действия для последующего обучения
    saveStateActionPair(
        agentState,
        replayBuffer,
        currentState,
        action,
        width,
        height,
    );

    // Исполнение действия (передача управляющих команд танку)
    executeAction(agentState.tankEid, action);

    return action;
}

// Исполнение действия
function executeAction(
    tankEid: number,
    action: ModelPrediction,
): void {
    TankController.setShooting$(tankEid, action.shoot);
    TankController.setMoveTarget$(tankEid, action.movement.x, action.movement.y);
    TankController.setTurretTarget$(tankEid, action.aim.x, action.aim.y);
}

// Сохранение текущего состояния и действия для обучения
function saveStateActionPair(
    agentState: TankState,
    replayBuffer: ReplayBuffer,
    state: Float32Array,
    action: ModelPrediction,
    width: number,
    height: number,
): void {
    // Если есть предыдущее состояние, рассчитать награду и добавить опыт в буфер
    if (agentState.lastState !== null && agentState.lastAction !== null) {
        const reward = calculateImprovedReward(agentState);

        // Проверка на завершение эпизода
        const isDone = TankInputTensor.health[agentState.tankEid] <= 0;

        // Добавление опыта в буфер воспроизведения
        addExperience(
            replayBuffer,
            agentState.lastState,
            agentState.lastAction,
            reward.totalReward,
            state,
            isDone,
        );

        // Сохранение истории наград для анализа
        agentState.episodeRewards.push(reward.totalReward);
        for (const [key, value] of Object.entries(reward.rewards)) {
            if (agentState.rewardHistory[key]) {
                agentState.rewardHistory[key].push(value);
            }
        }
    }

    // Обновление состояния и действия для следующего шага
    agentState.lastState = state;
    agentState.lastAction = {
        movement: [action.movement.x, action.movement.y],
        aim: [action.aim.x / width, action.aim.y / height], // нормализованные координаты
        shoot: action.shoot,
    };
    agentState.lastHealth = TankInputTensor.health[agentState.tankEid];

    // Обновление памяти о пулях
    updateBulletMemory(agentState);
}

// Улучшенная функция расчета награды с учетом истории
function calculateImprovedReward(agentState: TankState): RewardResult {
    const tankEid = agentState.tankEid;
    const tankX = TankInputTensor.x[tankEid];
    const tankY = TankInputTensor.y[tankEid];
    const currentHealth = TankInputTensor.health[tankEid];
    const tankSpeed = TankInputTensor.speed[tankEid];

    // Компоненты вознаграждения
    const rewardRecord: RewardComponents = {
        map: 0,
        aim: 0,
        avoidBullets: 0,
        avoidEnemies: 0,
        health: 0,
        damageDealt: 0,
        movement: 0,
        survival: 0,
    };

    // 1. Вознаграждение за пребывание в пределах карты
    if (tankX >= 0 && tankX <= window.innerWidth && tankY >= 0 && tankY <= window.innerHeight) {
        rewardRecord.map = 0;

        // Градиентный штраф при приближении к краю
        const borderDistance = Math.min(
            tankX,
            tankY,
            window.innerWidth - tankX,
            window.innerHeight - tankY,
        );

        if (borderDistance < 50) {
            // Линейный штраф за приближение к краю
            rewardRecord.map = (borderDistance / 50) * REWARD_WEIGHTS.BORDER_GRADIENT;
        }
    } else {
        // Штраф за выход за границы
        rewardRecord.map = REWARD_WEIGHTS.MAP_BORDER;
    }

    // 2. Вознаграждение за здоровье
    const healthChange = currentHealth - agentState.lastHealth;
    rewardRecord.health = healthChange * REWARD_WEIGHTS.HEALTH_CHANGE;

    // Бонус за оставшееся здоровье
    rewardRecord.health += currentHealth * REWARD_WEIGHTS.HEALTH_BONUS;

    // 3. Вознаграждение за прицеливание
    if (agentState.lastAction) {
        const targetAimX = agentState.lastAction.aim[0] * window.innerWidth;
        const targetAimY = agentState.lastAction.aim[1] * window.innerHeight;

        let bestAimQuality = 0;
        let hasTargets = false;
        let closestEnemyDist = Number.MAX_VALUE;

        // Оценка качества прицеливания
        for (let j = 0; j < TANK_INPUT_TENSOR_MAX_ENEMIES; j++) {
            // Учитываем ID врага (первым элементом)
            const enemyId = TankInputTensor.enemiesData.get(tankEid, j * 5); // ID врага
            const enemyX = TankInputTensor.enemiesData.get(tankEid, j * 5 + 1);
            const enemyY = TankInputTensor.enemiesData.get(tankEid, j * 5 + 2);

            if (enemyId !== 0) { // Проверяем по ID, а не по координатам
                hasTargets = true;
                const distToEnemy = Math.hypot(tankX - enemyX, tankY - enemyY);
                closestEnemyDist = Math.min(closestEnemyDist, distToEnemy);

                // Оценка точности прицеливания (насколько близко цель к врагу)
                const distFromTargetToEnemy = Math.hypot(targetAimX - enemyX, targetAimY - enemyY);
                const aimQuality = Math.max(0, 1 - distFromTargetToEnemy / (50 * 3)); // 50 - примерный радиус танка

                bestAimQuality = Math.max(bestAimQuality, aimQuality);
            }
        }

        rewardRecord.aim = bestAimQuality * REWARD_WEIGHTS.AIM_QUALITY;

        // Оценка стрельбы
        if (agentState.lastAction.shoot) {
            if (bestAimQuality > 0.7) {
                rewardRecord.aim += REWARD_WEIGHTS.SHOOTING_AIMED;
            } else if (bestAimQuality < 0.3) {
                rewardRecord.aim += REWARD_WEIGHTS.SHOOTING_RANDOM;
            }
        }

        // 4. Вознаграждение за поддержание дистанции
        if (hasTargets) {
            if (closestEnemyDist < 150) {
                // Слишком близко - опасно
                rewardRecord.avoidEnemies = -0.2;
            } else if (closestEnemyDist > 200 && closestEnemyDist < 500) {
                // Оптимальная дистанция
                rewardRecord.avoidEnemies = REWARD_WEIGHTS.DISTANCE_KEEPING;
            }
        }
    }

    // 5. Вознаграждение за избегание пуль (улучшенное)
    rewardRecord.avoidBullets = calculateBulletAvoidanceReward(agentState);

    // 6. Вознаграждение за движение
    rewardRecord.movement = 0;

    // Базовый бонус за движение
    if (tankSpeed > 50) {
        rewardRecord.movement += REWARD_WEIGHTS.MOVEMENT_BASE;
    }

    // Стратегическое движение в зависимости от ситуации
    if (rewardRecord.avoidBullets < -0.3 && tankSpeed > 200) {
        // Бонус за быстрое движение при опасности от пуль
        rewardRecord.movement += REWARD_WEIGHTS.STRATEGIC_MOVEMENT;
    }

    // 7. Бонус за выживание
    rewardRecord.survival = REWARD_WEIGHTS.SURVIVAL;

    // Дополнительный бонус при низком здоровье
    if (currentHealth < 0.3) {
        rewardRecord.survival += REWARD_WEIGHTS.SURVIVAL;
    }

    // Суммарное вознаграждение
    const totalReward =
        rewardRecord.map +
        rewardRecord.health +
        rewardRecord.aim +
        rewardRecord.avoidBullets +
        rewardRecord.avoidEnemies +
        rewardRecord.movement +
        rewardRecord.survival;

    // Ограничение экстремальных значений награды
    const clippedReward = Math.max(-15, Math.min(15, totalReward));

    return {
        totalReward: clippedReward,
        rewards: rewardRecord,
    };
}

// Обновление памяти о пулях
function updateBulletMemory(agentState: TankState): void {
    const currentTime = Date.now();
    const tankEid = agentState.tankEid;
    const tankX = TankInputTensor.x[tankEid];
    const tankY = TankInputTensor.y[tankEid];

    // Очистка устаревших записей (старше 2 секунд)
    for (const [bulletId, bulletInfo] of agentState.bulletMemory.entries()) {
        if (currentTime - bulletInfo.timestamp > 2000) {
            agentState.bulletMemory.delete(bulletId);
        }
    }

    // Обновление существующих записей и добавление новых
    for (let j = 0; j < TANK_INPUT_TENSOR_MAX_BULLETS; j++) {
        // Учитываем, что первое значение - это ID пули
        const bulletId = TankInputTensor.bulletsData.get(tankEid, j * 5);
        const bulletX = TankInputTensor.bulletsData.get(tankEid, j * 5 + 1);
        const bulletY = TankInputTensor.bulletsData.get(tankEid, j * 5 + 2);
        const bulletVx = TankInputTensor.bulletsData.get(tankEid, j * 5 + 3);
        const bulletVy = TankInputTensor.bulletsData.get(tankEid, j * 5 + 4);

        // Пропуск несуществующих или статичных пуль
        if (bulletId === 0 || Math.hypot(bulletVx, bulletVy) < 100) continue;

        // Создание уникального идентификатора пули используя её ID
        const uniqueBulletId = `bullet_${ bulletId }`;

        // Проверка, если пуля уже отслеживается
        if (agentState.bulletMemory.has(uniqueBulletId)) continue;

        // Добавление новой пули в память
        agentState.bulletMemory.set(uniqueBulletId, {
            id: bulletId,
            x: bulletX,
            y: bulletY,
            vx: bulletVx,
            vy: bulletVy,
            timestamp: currentTime,
            wasAvoided: false,
        });
    }

    // Проверка, были ли успешно избежаны пули
    for (const [bulletId, bulletInfo] of agentState.bulletMemory.entries()) {
        // Если пуля еще не была отмечена как избежанная
        if (!bulletInfo.wasAvoided) {
            // Рассчитать текущее положение пули на основе прошедшего времени
            const elapsedTime = (currentTime - bulletInfo.timestamp) / 1000; // в секундах
            const estimatedX = bulletInfo.x + bulletInfo.vx * elapsedTime;
            const estimatedY = bulletInfo.y + bulletInfo.vy * elapsedTime;

            // Расстояние между пулей и танком
            const distance = Math.hypot(estimatedX - tankX, estimatedY - tankY);

            // Если пуля прошла мимо танка (рассчитываем на основе скорости и направления)
            const bulletSpeed = Math.hypot(bulletInfo.vx, bulletInfo.vy);
            const bulletDirectionX = bulletInfo.vx / bulletSpeed;
            const bulletDirectionY = bulletInfo.vy / bulletSpeed;

            // Вектор от начального положения пули к танку
            const toTankX = tankX - bulletInfo.x;
            const toTankY = tankY - bulletInfo.y;

            // Скалярное произведение для определения, прошла ли пуля мимо танка
            const dotProduct = toTankX * bulletDirectionX + toTankY * bulletDirectionY;

            // Если пуля прошла мимо танка и на безопасном расстоянии
            if (dotProduct < 0 && distance > 100) {
                bulletInfo.wasAvoided = true;
                agentState.bulletMemory.set(bulletId, bulletInfo);
            }
        }
    }
}

// Расчет награды за избегание пуль на основе памяти о пулях
function calculateBulletAvoidanceReward(agentState: TankState): number {
    let reward = 0;
    const tankEid = agentState.tankEid;
    const tankX = TankInputTensor.x[tankEid];
    const tankY = TankInputTensor.y[tankEid];
    const tankSpeed = TankInputTensor.speed[tankEid];

    // Подсчет количества успешно избежанных пуль
    let avoidedBullets = 0;
    let dangerousBullets = 0;

    for (const [_, bulletInfo] of agentState.bulletMemory.entries()) {
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
                        dangerousBullets++;
                        reward -= dangerLevel * REWARD_WEIGHTS.BULLET_AVOIDANCE;

                        // Уменьшаем штраф, если танк активно движется
                        if (tankSpeed > 200) {
                            reward += dangerLevel * REWARD_WEIGHTS.BULLET_AVOIDANCE * 0.5;
                        }
                    }
                }
            }
        }
    }

    // Дополнительный бонус за отсутствие опасных пуль при наличии избежанных
    if (dangerousBullets === 0 && avoidedBullets > 0) {
        reward += avoidedBullets * 0.2;
    }

    return reward;
}

// Получить статистику обучения
export function getAgentStats(agentState: TankState): AgentStats {
    return {
        episodeRewards: agentState.episodeRewards,
        rewardHistory: agentState.rewardHistory,
        bulletMemory: {
            count: agentState.bulletMemory.size,
            avoided: Array.from(agentState.bulletMemory.values()).filter(b => b.wasAvoided).length,
        },
    };
}

// Сброс агента для новой игры
export function resetAgent(agentState: TankState): TankState {
    return {
        ...agentState,
        lastState: null,
        lastAction: null,
        lastHealth: 1.0,
        bulletMemory: new Map(),
    };
}