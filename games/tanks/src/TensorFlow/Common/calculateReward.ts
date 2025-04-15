import { RigidBodyState } from '../../ECS/Components/Physical.ts';
import { BattleState, BULLET_DANGER_SPEED, getBattleState } from '../../ECS/Systems/RL/createTankInputTensorSystem.ts';
import { abs, centerStep, hypot, lerp, max, min, smoothstep } from '../../../../../lib/math.ts';
import { TANK_RADIUS } from './consts.ts';
import { getMatrixTranslation, LocalTransform } from '../../../../../src/ECS/Components/Transform.ts';
import { Tank } from '../../ECS/Components/Tank/Tank.ts';
import { TankController } from '../../ECS/Components/Tank/TankController.ts';
import { CONFIG } from '../PPO/config.ts';
import { ALLY_BUFFER, BULLET_BUFFER, ENEMY_BUFFER, TankInputTensor } from '../../ECS/Components/Tank/TankState.ts';
import { getTankHealth } from '../../ECS/Components/Tank/TankHealth.ts';

// Константы для калибровки вознаграждений
let REWARD_WEIGHTS = {
    TEAM: {
        HEALTH: 5, // за разницу в здоровье, может быть отрицательной
        KILLS: 5, // за разницу в количестве убитых врагов, может быть отрицательной
    },
    TEAM_MULTIPLIER: 0.5,

    COMMON: {
        HEALTH: 1, // За хороший показатель здоровья
        SURVIVAL: 1, // За выживание в бою
    },
    COMMON_MULTIPLIER: 0.5,

    AIM: {
        QUALITY: 1.0,       // За точное прицеливание
        DISTANCE: 0.1,      // За расстояние до прицела
        NO_TARGET_PENALTY: -0.1, // За отсутствие целей
        DISTANCE_PENALTY: -0.1, // За расстояние до цели
        SHOOTING: 1.0,       // За стрельбу в цель
        SHOOTING_PENALTY: -0.1, // За стрельбу в пустоту
        SHOOTING_ALLIES_PENALTY: -3.0, // За стрельбу в союзников
    },
    AIM_MULTIPLIER: 3,

    MAP_BORDER: {
        BASE: 1,          // За нахождение в пределах карты
        RETURN: 1,        // За возвращение в пределы карты
        PENALTY: -2.5,    // За выход за границы
    },
    MAP_BORDER_MULTIPLIER: 1,

    DISTANCE_KEEPING: {
        BASE: 1.0,          // За поддержание дистанции
        PENALTY: -2.5,      // За слишком близкое приближение
    },
    DISTANCE_KEEPING_MULTIPLIER: 1, // может быть несколько врагов

    MOVEMENT: {
        BASE: 0.2,          // За базовое движение
        STRATEGIC: 0.4,     // За стратегическое движение
    },
    MOVEMENT_MULTIPLIER: 1,

    BULLET_AVOIDANCE: {
        PENALTY: -0.8,
        AVOID_QUALITY: 0.8,
    },
    BULLET_AVOIDANCE_MULTIPLIER: 1,
};

// Структура для хранения многокомпонентных наград
export interface ComponentRewards {
    team: {
        health: number;
        kills: number;
        total: number;
    };

    common: {
        health: number;
        survival: number;
        total: number;
    };

    aim: {
        distance: number;        // Расстояние до цели
        accuracy: number;        // Точность прицеливания
        shootDecision: number;   // Решение о стрельбе
        total: number;           // Суммарная награда для головы прицеливания
    };

    // Награды для головы движения
    positioning: {
        movement: number;
        enemiesPositioning: number;     // Позиционирование относительно врагов
        alliesPositioning: number;      // Позиционирование относительно союзников
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
        team: { health: 0, kills: 0, total: 0 },
        aim: { accuracy: 0, distance: 0, total: 0, shootDecision: 0 },
        positioning: {
            movement: 0,
            enemiesPositioning: 0,
            alliesPositioning: 0,
            bulletAvoidance: 0,
            mapAwareness: 0,
            total: 0,
        },
        totalReward: 0,
    };
}

const EPSILON = 1e-6; // Для избежания деления на ноль

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
    // const [beforePredictTurretTargetX, beforePredictTurretTargetY] = TankInputTensor.turretTarget.getBatch(tankEid);
    const beforePredictEnemiesData = TankInputTensor.enemiesData.getBatch(tankEid);
    const beforePredictAlliesData = TankInputTensor.alliesData.getBatch(tankEid);
    const beforePredictBulletsData = TankInputTensor.bulletsData.getBatch(tankEid);
    const beforePredictBattleState: BattleState = {
        alliesCount: TankInputTensor.alliesCount[tankEid],
        alliesTotalHealth: TankInputTensor.alliesTotalHealth[tankEid],
        enemiesCount: TankInputTensor.enemiesCount[tankEid],
        enemiesTotalHealth: TankInputTensor.enemiesTotalHealth[tankEid],
    };

    // -- current state --
    const moveDir = TankController.move[tankEid];
    const isShooting = TankController.shoot[tankEid] > 0;
    const currentHealth = getTankHealth(tankEid);
    const [currentTankX, currentTankY] = RigidBodyState.position.getBatch(tankEid);
    // const [currentTankSpeedX, currentTankSpeedY] = RigidBodyState.linvel.getBatche(tankEid);
    const [currentTurretTargetX, currentTurretTargetY] = getMatrixTranslation(LocalTransform.matrix.getBatch(Tank.aimEid[tankEid]));
    // const currentShootings = TankController.shoot[tankEid] > 0;
    // const currentEnemies = findTankEnemies(tankEid);
    // const currentDangerBullets = findTankDangerBullets(tankEid);
    const currentBattleState = getBattleState(tankEid);

    // Инициализируем пустую структуру наград
    const rewards = initializeRewards();

    // 0. Награда за здоровье команды
    rewards.team.health = REWARD_WEIGHTS.TEAM.HEALTH * calculateTeamHealth(beforePredictBattleState, currentBattleState);
    rewards.team.kills = REWARD_WEIGHTS.TEAM.KILLS * calculateTeamKills(beforePredictBattleState, currentBattleState);

    // 1. Награда за выживание
    rewards.common.health = REWARD_WEIGHTS.COMMON.HEALTH * currentHealth;
    rewards.common.survival = REWARD_WEIGHTS.COMMON.SURVIVAL * (step / CONFIG.episodeFrames);

    // 2. Расчет награды за нахождение в пределах карты
    rewards.positioning.mapAwareness = calculateTankMapAwarenessReward(
        width,
        height,
        currentTankX,
        currentTankY,
        beforePredictTankX,
        beforePredictTankY,
    );

    // 3. Анализ целей и вычисление награды за прицеливание
    const aimingResult = analyzeAiming(
        currentTankX,
        currentTankY,
        currentTurretTargetX,
        currentTurretTargetY,
        beforePredictEnemiesData,
        beforePredictAlliesData,
    );

    rewards.aim.accuracy = aimingResult.aimQualityReward;
    rewards.aim.distance = aimingResult.aimDistanceReward;

    // 5. Награда за решение о стрельбе
    rewards.aim.shootDecision = calculateShootingReward(
        isShooting,
        aimingResult.bestEnemyAimQuality,
        aimingResult.sumAlliesAimQuality,
    );

    // 6. Награда за избегание пуль
    const bulletAvoidanceResult = calculateBulletAvoidanceReward(
        currentTankX,
        currentTankY,
        beforePredictBulletsData,
    );
    rewards.positioning.bulletAvoidance = bulletAvoidanceResult.reward;

    // 7. Награда за движение и позиционирование
    rewards.positioning.movement = calculateMovementReward(
        moveDir,
        bulletAvoidanceResult.maxDangerLevel,
    );
    rewards.positioning.enemiesPositioning = calculateEnemyDistanceReward(
        currentTankX,
        currentTankY,
        beforePredictEnemiesData,
    );

    rewards.positioning.alliesPositioning = calculateAllyDistanceReward(
        currentTankX,
        currentTankY,
        beforePredictAlliesData,
    );

    // Рассчитываем итоговые значения
    rewards.team.total = REWARD_WEIGHTS.TEAM_MULTIPLIER * (rewards.team.health + rewards.team.kills);
    rewards.common.total = REWARD_WEIGHTS.COMMON_MULTIPLIER
        * (rewards.common.health + rewards.common.survival);
    rewards.aim.total = REWARD_WEIGHTS.AIM_MULTIPLIER
        * (rewards.aim.accuracy + rewards.aim.distance + rewards.aim.shootDecision);
    rewards.positioning.total =
        (rewards.positioning.movement * REWARD_WEIGHTS.MOVEMENT_MULTIPLIER
            + rewards.positioning.enemiesPositioning * REWARD_WEIGHTS.DISTANCE_KEEPING_MULTIPLIER
            + rewards.positioning.alliesPositioning * REWARD_WEIGHTS.DISTANCE_KEEPING_MULTIPLIER
            + rewards.positioning.bulletAvoidance * REWARD_WEIGHTS.BULLET_AVOIDANCE_MULTIPLIER
            + rewards.positioning.mapAwareness * REWARD_WEIGHTS.MAP_BORDER_MULTIPLIER);

    // Общая итоговая награда
    rewards.totalReward = rewards.team.total + rewards.common.total + rewards.aim.total + rewards.positioning.total;

    return rewards;
}

function calculateTeamHealth(prevBattleState: BattleState, currentBattleState: BattleState): number {
    const alliesDelta = (currentBattleState.alliesTotalHealth - prevBattleState.alliesTotalHealth)
        / max(1, currentBattleState.alliesCount, prevBattleState.alliesCount);
    const enemiesDelta = (currentBattleState.enemiesTotalHealth - prevBattleState.enemiesTotalHealth)
        / max(1, currentBattleState.enemiesCount, prevBattleState.enemiesCount);
    return alliesDelta - enemiesDelta;
}

function calculateTeamKills(prevBattleState: BattleState, currentBattleState: BattleState): number {
    const alliesDelta = (currentBattleState.alliesCount - prevBattleState.alliesCount);
    const enemiesDelta = (currentBattleState.enemiesCount - prevBattleState.enemiesCount);
    return alliesDelta - enemiesDelta;
}

function calculateTankMapAwarenessReward(
    width: number,
    height: number,
    x: number,
    y: number,
    prevX: number,
    prevY: number,
): number {
    const isInBounds = x >= 0 && x <= width && y >= 0 && y <= height;
    const borderDistance = min(
        x,
        y,
        width - x,
        height - y,
    );

    if (isInBounds) {
        return REWARD_WEIGHTS.MAP_BORDER.BASE;
    } else {
        let reward = REWARD_WEIGHTS.MAP_BORDER.PENALTY * smoothstep(0, -100, borderDistance);

        // Вычисляем вектор движения танка
        const dx = x - prevX;
        const dy = y - prevY;

        // Длина вектора движения (для нормализации)
        const moveLength = Math.sqrt(dx * dx + dy * dy);

        // Если танк не двигался, просто возвращаем штраф
        if (moveLength < 0.1) {
            return reward;
        }

        // Определяем ближайшую точку границы карты для возврата
        let targetX = 0;
        let targetY = 0;

        // Корректируем целевую точку к ближайшей границе
        if (x < 0) targetX = 0;
        else if (x > width) targetX = width;
        else targetX = x;

        if (y < 0) targetY = 0;
        else if (y > height) targetY = height;
        else targetY = y;

        // Вектор направления к ближайшей точке карты
        const toMapX = targetX - prevX;
        const toMapY = targetY - prevY;

        // Длина вектора к карте
        const toMapLength = hypot(toMapX, toMapY);

        // Нормализуем векторы
        const normalizedDx = dx / moveLength;
        const normalizedDy = dy / moveLength;
        const normalizedToMapX = toMapX / toMapLength;
        const normalizedToMapY = toMapY / toMapLength;

        // Скалярное произведение векторов для определения косинуса угла между ними
        // Значение от -1 (противоположные направления) до 1 (совпадающие направления)
        const dotProduct = normalizedDx * normalizedToMapX + normalizedDy * normalizedToMapY;

        // Преобразуем в диапазон [0, 1], где 1 означает движение прямо к карте
        const directionAlignment = (dotProduct + 1) / 2;

        // Дополнительно учитываем, приближаемся ли мы к карте или удаляемся от неё
        const distanceToBorderBefore = distanceToMap(width, height, prevX, prevY);
        const distanceToBorderAfter = distanceToMap(width, height, x, y);

        // Если мы приближаемся к карте, даём дополнительный бонус
        if (distanceToBorderAfter < distanceToBorderBefore) {
            const directionBonus = REWARD_WEIGHTS.MAP_BORDER.RETURN * directionAlignment;
            reward += directionBonus;
        }

        return reward;
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
    turretTargetX: number,
    turretTargetY: number,
    beforePredictEnemiesData: Float64Array,
    beforePredictAlliesData: Float64Array,
): {
    bestEnemyAimQuality: number;
    bestEnemyAimTargetId: number;
    sumAlliesAimQuality: number;
    aimQualityReward: number;
    aimDistanceReward: number;
} {
    let bestEnemyTangentialAimDist = 0;
    let bestEnemyTangentialAimQuality = 0;

    let bestEnemyAimQuality = 0;
    let bestEnemyAimTargetId = 0;

    let sumAlliesAimQuality = 0;

    // Анализируем всех видимых врагов для текущего состояния
    for (let i = 0; i < beforePredictEnemiesData.length; i += ENEMY_BUFFER) {
        const enemyId = beforePredictEnemiesData[i];

        if (enemyId === 0) continue;

        const enemyX = RigidBodyState.position.get(enemyId, 0);
        const enemyY = RigidBodyState.position.get(enemyId, 1);

        const {
            quality: currentAimQuality,
            tangentialAimQuality: currentTangentialAimQuality,
        } = computeAimQuality(tankX, tankY, turretTargetX, turretTargetY, enemyX, enemyY);

        // Отслеживаем лучшее прицеливание
        if (currentAimQuality > bestEnemyAimQuality) {
            bestEnemyAimQuality = currentAimQuality;
            bestEnemyAimTargetId = enemyId;
        }

        if (currentTangentialAimQuality > bestEnemyTangentialAimQuality) {
            bestEnemyTangentialAimDist = hypot(tankX - enemyX, tankY - enemyY);
            bestEnemyTangentialAimQuality = currentTangentialAimQuality;
        }
    }

    // Анализируем всех видимых союзников для текущего состояния
    for (let i = 0; i < beforePredictAlliesData.length; i += ALLY_BUFFER) {
        const allyId = beforePredictAlliesData[i];

        if (allyId === 0) continue;

        const allyX = RigidBodyState.position.get(allyId, 0);
        const allyY = RigidBodyState.position.get(allyId, 1);
        const dist = hypot(tankX - allyX, tankY - allyY);
        let {
            quality,
            tangentialAimQuality,
        } = computeAimQuality(tankX, tankY, turretTargetX, turretTargetY, allyX, allyY);

        if (dist > bestEnemyTangentialAimDist) {
            const chanceToMissEnemy = 1 - bestEnemyTangentialAimQuality;
            const chanceToHitAlly = min(chanceToMissEnemy, tangentialAimQuality);
            quality *= chanceToHitAlly;
        }

        sumAlliesAimQuality += quality;
    }

    // Награда за качество прицеливания и дистанцию до цели
    const aimQualityReward =
        (bestEnemyAimQuality * REWARD_WEIGHTS.AIM.QUALITY)
        + (bestEnemyAimTargetId === 0 ? REWARD_WEIGHTS.AIM.NO_TARGET_PENALTY : 0);

    // Награда за дистанцию прицеливания
    const turretTargetDistance = hypot(turretTargetX - tankX, turretTargetY - tankY);
    const aimDistanceReward =
        REWARD_WEIGHTS.AIM.DISTANCE * (
            turretTargetDistance < 300
                ? smoothstep(TANK_RADIUS, 300, turretTargetDistance)
                : smoothstep(800, 300, turretTargetDistance)
        )
        + REWARD_WEIGHTS.AIM.DISTANCE_PENALTY * smoothstep(TANK_RADIUS, 0, turretTargetDistance)
        + REWARD_WEIGHTS.AIM.DISTANCE_PENALTY * smoothstep(800, 1000, turretTargetDistance);

    return {
        bestEnemyAimQuality,
        bestEnemyAimTargetId,
        aimQualityReward,
        aimDistanceReward,
        sumAlliesAimQuality,
    };
}

function computeAimQuality(
    tankX: number,
    tankY: number,
    turretX: number,
    turretY: number,
    enemyX: number,
    enemyY: number,
): { quality: number; tangentialAimQuality: number } {
    // Вектор от танка к турели
    const tankToTurretX = turretX - tankX;
    const tankToTurretY = turretY - tankY;

    // Вектор от танка к противнику
    const tankToEnemyX = enemyX - tankX;
    const tankToEnemyY = enemyY - tankY;

    const turretToEnemyX = enemyX - turretX;
    const turretToEnemyY = enemyY - turretY;

    // Вычисляем длины векторов
    const tankToTurretDist = hypot(tankToTurretX, tankToTurretY);
    const tankToEnemyDist = hypot(tankToEnemyX, tankToEnemyY);
    const turretToEnemyDist = hypot(turretToEnemyX, turretToEnemyY);

    // Учитываем расстояние до противника (как в исходной функции)
    const distanceQuality =
        lerp(0, 0.02, smoothstep(1000, 0, turretToEnemyDist))
        + lerp(0, 0.02, smoothstep(TANK_RADIUS * 3, 0, turretToEnemyDist))
        + lerp(0, 0.96, smoothstep(TANK_RADIUS * 1.2, 0, turretToEnemyDist));

    // Нормализованные векторы
    const turretNormX = tankToTurretX / (tankToTurretDist + EPSILON);
    const turretNormY = tankToTurretY / (tankToTurretDist + EPSILON);
    const enemyNormX = tankToEnemyX / (tankToEnemyDist + EPSILON);
    const enemyNormY = tankToEnemyY / (tankToEnemyDist + EPSILON);

    // Скалярное произведение нормализованных векторов
    const dotProduct = turretNormX * enemyNormX + turretNormY * enemyNormY;

    // Угол между векторами (в радианах)
    const angle = Math.acos(Math.max(-1, Math.min(1, dotProduct)));

    // Вычисляем векторное произведение для определения знака (по часовой или против)
    const crossProduct = turretNormX * enemyNormY - turretNormY * enemyNormX;
    const signedAngle = crossProduct >= 0 ? angle : -angle;

    // Вычисляем расстояние, на котором линия выстрела пройдет от противника (расстояние касательной)

    // Вычисляем перпендикулярное расстояние от линии выстрела до противника (это и есть расстояние касательной)
    const tangentialDistance = Math.sin(angle) * tankToEnemyDist;

    // Вычисляем качество прицеливания для прямого выстрела
    // Чем меньше угол, тем лучше прицеливание для прямого выстрела
    const directAimQuality = smoothstep(Math.PI / 4, 0, Math.abs(signedAngle));

    // Вычисляем качество прицеливания для касательного выстрела
    // Награда за выстрел, проходящий на оптимальном расстоянии от противника
    const tangentialAimQuality = smoothstep(TANK_RADIUS * 1.5, 0, Math.abs(tangentialDistance));

    return {
        quality: 0.2 * distanceQuality + 0.3 * directAimQuality + 0.5 * tangentialAimQuality,
        tangentialAimQuality,
    };
}

/**
 * Расчет награды за решение о стрельбе
 */
function calculateShootingReward(
    isShooting: boolean,
    bestEnemyAimQuality: number,
    sumAlliesAimQuality: number,
): number {
    let shootingReward = 0;

    if (isShooting && sumAlliesAimQuality > bestEnemyAimQuality) {
        return REWARD_WEIGHTS.AIM.SHOOTING_ALLIES_PENALTY;
    }

    if (isShooting) {
        // Плавная награда за стрельбу в зависимости от точности прицеливания
        shootingReward += bestEnemyAimQuality * REWARD_WEIGHTS.AIM.SHOOTING;
    } else if (bestEnemyAimQuality > 0.7) {
        // Небольшой штраф за отсутствие стрельбы при хорошем прицеливании
        shootingReward += REWARD_WEIGHTS.AIM.SHOOTING_PENALTY * smoothstep(0.8, 1.0, bestEnemyAimQuality);
    }

    return shootingReward;
}

/**
 * Расчет награды за движение
 */
function calculateMovementReward(
    moveDir: number,
    maxDangerLevel: number,
): number {
    let speedReward = 0;
    const moveFactor = abs(moveDir);

    // Базовая награда за движение
    speedReward += moveFactor * REWARD_WEIGHTS.MOVEMENT.BASE;

    // Стратегическое движение при наличии опасности
    if (maxDangerLevel > 0) {
        // Дополнительная награда за движение при наличии опасных пуль
        speedReward += moveFactor * REWARD_WEIGHTS.MOVEMENT.STRATEGIC;
    }

    return speedReward;
}

function calculateEnemyDistanceReward(
    tankX: number,
    tankY: number,
    beforePredictEnemiesData: Float64Array,
): number {
    let positioningReward = 0;

    // Анализируем всех видимых врагов для текущего состояния
    for (let i = 0; i < beforePredictEnemiesData.length; i += ENEMY_BUFFER) {
        const enemyId = beforePredictEnemiesData[i];

        if (enemyId === 0) continue;

        const enemyX = RigidBodyState.position.get(enemyId, 0);
        const enemyY = RigidBodyState.position.get(enemyId, 1);

        const distToEnemy = hypot(tankX - enemyX, tankY - enemyY);

        if (distToEnemy < TANK_RADIUS * 3) {
            // Штраф за слишком близкое расстояние
            const tooClosePenalty = 1 - smoothstep(0, TANK_RADIUS * 3, distToEnemy);
            positioningReward += tooClosePenalty * REWARD_WEIGHTS.DISTANCE_KEEPING.PENALTY;
        } else if (distToEnemy <= 800) {
            // Награда за оптимальную дистанцию
            const optimalDistanceReward = lerp(0.3, 1, centerStep(TANK_RADIUS * 3, 800, distToEnemy));
            positioningReward += optimalDistanceReward * REWARD_WEIGHTS.DISTANCE_KEEPING.BASE;
        } else {
            // Мягкий штраф за слишком большую дистанцию
            const tooFarPenalty = smoothstep(800, 1200, distToEnemy) * REWARD_WEIGHTS.DISTANCE_KEEPING.PENALTY / 5;
            positioningReward += tooFarPenalty;
        }
    }

    return positioningReward;
}

function calculateAllyDistanceReward(
    tankX: number,
    tankY: number,
    beforePredictAlliesData: Float64Array,
): number {
    let positioningReward = 0;

    // Анализируем всех видимых врагов для текущего состояния
    for (let i = 0; i < beforePredictAlliesData.length; i += ALLY_BUFFER) {
        const allyId = beforePredictAlliesData[i];

        if (allyId === 0) continue;

        const allyX = RigidBodyState.position.get(allyId, 0);
        const allyY = RigidBodyState.position.get(allyId, 1);

        const distToAlly = hypot(tankX - allyX, tankY - allyY);

        if (distToAlly < TANK_RADIUS * 3) {
            // Штраф за слишком близкое расстояние
            const tooClosePenalty = 1 - smoothstep(0, TANK_RADIUS * 3, distToAlly);
            positioningReward += tooClosePenalty * REWARD_WEIGHTS.DISTANCE_KEEPING.PENALTY;
        }
    }

    return positioningReward;
}

/**
 * Расчет награды за избегание пуль
 */
function calculateBulletAvoidanceReward(
    tankX: number,
    tankY: number,
    beforePredictBulletsData: Float64Array,
): { reward: number; maxDangerLevel: number } {
    let reward = 0;
    let maxDangerLevel = 0;

    // Анализируем каждую опасную пулю
    for (let i = 0; i < beforePredictBulletsData.length; i += BULLET_BUFFER) {
        const bulletId = beforePredictBulletsData[i];

        if (bulletId === 0) continue;

        const bulletX = RigidBodyState.position.get(bulletId, 0);
        const bulletY = RigidBodyState.position.get(bulletId, 1);
        const bulletVx = RigidBodyState.linvel.get(bulletId, 0);
        const bulletVy = RigidBodyState.linvel.get(bulletId, 1);

        // Анализируем пулю с учетом как текущего, так и предыдущего состояния
        const bulletResult = analyzeBullet(
            tankX,
            tankY,
            bulletX,
            bulletY,
            bulletVx,
            bulletVy,
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
): { reward: number; dangerLevel: number } {
    // Анализируем текущую опасность пули
    const bulletSpeed = hypot(bulletVx, bulletVy);

    if (bulletSpeed < BULLET_DANGER_SPEED) return { reward: 0, dangerLevel: 0 };

    const bulletDirectionX = bulletVx / bulletSpeed;
    const bulletDirectionY = bulletVy / bulletSpeed;

    // Вектор от пули к танку
    const toTankX = tankX - bulletX;
    const toTankY = tankY - bulletY;

    // Определяем, движется ли пуля к танку
    const dotProduct = toTankX * bulletDirectionX + toTankY * bulletDirectionY;

    if (dotProduct < 0) return { reward: 0, dangerLevel: 0 };

    // Точка ближайшего прохождения пули к танку
    const closestPointX = bulletX + bulletDirectionX * dotProduct;
    const closestPointY = bulletY + bulletDirectionY * dotProduct;

    // Расстояние в точке наибольшего сближения
    const minDist = hypot(closestPointX - tankX, closestPointY - tankY);

    const dangerLevel = smoothstep(TANK_RADIUS * 1.5, TANK_RADIUS / 2, minDist);

    const reward = dangerLevel < 0.2
        ? (1 - dangerLevel) * REWARD_WEIGHTS.BULLET_AVOIDANCE.AVOID_QUALITY
        : dangerLevel * REWARD_WEIGHTS.BULLET_AVOIDANCE.PENALTY;

    return { reward, dangerLevel };
}