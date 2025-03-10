import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';
import { getTankHealth } from '../../ECS/Components/Tank.ts';
import { RigidBodyState } from '../../ECS/Components/Physical.ts';
import { TankController } from '../../ECS/Components/TankController.ts';
import { findTankDangerBullets, findTankEnemies } from '../../ECS/Systems/createTankInputTensorSystem.ts';

// Константы для калибровки вознаграждений
export const REWARD_WEIGHTS = {
    // Основные компоненты наград
    HEALTH_CHANGE: 0.5,          // За потерю здоровья
    HEALTH_BONUS: 0.05,          // За поддержание здоровья
    SURVIVAL: 0.05,              // За выживание

    MOVEMENT_BASE: 0.1,          // За базовое движение
    BULLET_AVOIDANCE: 0.4,      // За попадание под пули
    STRATEGIC_MOVEMENT: 0.3,     // За стратегическое движение

    MAP_BORDER: -1.0,            // За выход за границы
    BORDER_GRADIENT: -0.5,       // За приближение к границе
    DISTANCE_KEEPING: 0.5,       // За поддержание дистанции

    SHOOTING_AIMED: 2.0,         // За прицельную стрельбу
    SHOOTING_RANDOM: -0.2,       // За беспорядочную стрельбу

    AIM_QUALITY: 2.0,            // За точное прицеливание
    AIM_DISTANCE: -1.0,            // За точное прицеливание
    AIM_TRACKING: 2.0,           // За активное отслеживание врага
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
        distance: number;    // Нахождение в пределах карты
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

export function calculateNewReward(
    tankEid: number,
    actions: ArrayLike<number>,
    width: number,
    height: number,
): ComponentRewards {
    // before predict
    const beforePredictHealth = TankInputTensor.health[tankEid];
    const beforePredictTankXY = TankInputTensor.position.getBatche(tankEid);
    const beforePredictTankSpeedXY = TankInputTensor.speed.getBatche(tankEid);
    const beforePridectTurretTargetXY = TankInputTensor.turretTarget.getBatche(tankEid);
    const beforePredictBulletsData = TankInputTensor.bulletsData.getBatche(tankEid);
    const beforePredictEnemiesData = TankInputTensor.enemiesData.getBatche(tankEid);
    // current state
    const currentHealth = getTankHealth(tankEid);
    const currentTankXY = RigidBodyState.position.getBatche(tankEid);
    const currentTankSpeedXY = RigidBodyState.linvel.getBatche(tankEid);
    const currentTurretTargetXY = TankController.turretTarget.getBatche(tankEid);
    const currentShootings = TankController.shoot[tankEid] > 0;
    const [currentEnemiesCount, currentEnemiesList] = findTankEnemies(tankEid);
    const [currentDangerBulletsCount, currentDangerBulletsList] = findTankDangerBullets(tankEid);

    const isShooting = actions[0] > 0.5; // выстрел
    const moveDirAction = actions[1] || 0; // forward/backward
    const moveRotAction = actions[2] || 0; // rotate left/right
    const aimDeltaXAction = actions[3] || 0;  // change aim by X[-1,1], from beforePredictTurretTargetXY.x
    const aimDeltaYAction = actions[4] || 0;  // change aim by Y[-1,1], from beforePredictTurretTargetXY.y

    // iterate enemies/bullets
    // const id = bulletsList[i];
    // const x = RigidBodyState.position.get(id, 0);
    // const y = RigidBodyState.position.get(id, 1);
    // const vx = RigidBodyState.linvel.get(id, 0);
    // const vy = RigidBodyState.linvel.get(id, 1);

    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_BULLETS; i++) {
        const id = TankInputTensor.bulletsData.get(tankEid, i * 5);
        const x = TankInputTensor.bulletsData.get(tankEid, i * 5 + 1);
        const y = TankInputTensor.bulletsData.get(tankEid, i * 5 + 2);
        const vx = TankInputTensor.bulletsData.get(tankEid, i * 5 + 3);
        const vy = TankInputTensor.bulletsData.get(tankEid, i * 5 + 4);

        if (id === 0) ;
    }

    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_ENEMIES; i++) {
        const id = TankInputTensor.enemiesData.get(tankEid, i * 5);
        const x = TankInputTensor.enemiesData.get(tankEid, i * 5 + 1);
        const y = TankInputTensor.enemiesData.get(tankEid, i * 5 + 2);
        const vx = TankInputTensor.enemiesData.get(tankEid, i * 5 + 3);
        const vy = TankInputTensor.enemiesData.get(tankEid, i * 5 + 4);

        if (id === 0) ;
    }
}
