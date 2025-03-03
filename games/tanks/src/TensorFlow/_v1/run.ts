import { createBattlefield } from '../Common/createBattlefield.ts';
import { MAX_STEPS, TANK_COUNT_SIMULATION, TICK_TIME_REAL, TICK_TIME_SIMULATION } from '../Common/consts.ts';
import { query } from 'bitecs';
import { Tank } from '../../ECS/Components/Tank.ts';

import {
    createTankManager,
    disposeTankManager,
    getTrainingStats,
    registerTank,
    startTraining,
    stopTraining,
    TankManagerState,
    unregisterTank,
    updateTankManager,
} from './manager';
import { TankInputTensor } from '../../ECS/Components/TankState';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';

// Константы игры
const GAME_WIDTH = 800;
const GAME_HEIGHT = 800;
const MAX_SPEED = 1000;

// Пример интеграции с основной игрой
export async function initializeGameAI(): Promise<TankManagerState> {
    try {
        // Создание менеджера танков
        const managerState = await createTankManager(
            GAME_WIDTH,
            GAME_HEIGHT,
            MAX_SPEED,
        );

        console.log('Tank AI system initialized successfully');
        return managerState;
    } catch (error) {
        console.error('Error initializing tank AI system:', error);
        throw error;
    }
}

// Добавление AI контроля к танку
export function addControlToTank(
    managerState: TankManagerState,
    tankEntityId: number,
): TankManagerState {
    // Проверка, что танк существует и имеет необходимые компоненты
    // if (!TankController.has(tankEntityId) || !TankInputTensor.exists(tankEntityId)) {
    //     console.error(`Tank entity ${tankEntityId} is missing required components`);
    //     return managerState;
    // }

    // Регистрация танка в менеджере
    const updatedManager = registerTank(managerState, tankEntityId);
    console.log(`AI control added to tank ${ tankEntityId }`);

    return updatedManager;
}

// Удаление AI контроля с танка
export function removeAIControlFromTank(
    managerState: TankManagerState,
    tankEntityId: number,
): TankManagerState {
    const updatedManager = unregisterTank(managerState, tankEntityId);
    console.log(`AI control removed from tank ${ tankEntityId }`);

    return updatedManager;
}

// Функция обновления на каждом игровом тике
export function updateGameAI(managerState: TankManagerState): TankManagerState {
    // Удаление уничтоженных танков
    let updatedManager = { ...managerState };

    for (const tankId of managerState.agents.keys()) {
        if (
            // !TankController.has(tankId) ||
            // !TankInputTensor.exists(tankId) ||
            TankInputTensor.health[tankId] <= 0
        ) {
            debugger
            updatedManager = unregisterTank(updatedManager, tankId);
        }
    }

    // Обновление менеджера танков
    return updateTankManager(updatedManager);
}

// Начать обучение AI
export function startAITraining(managerState: TankManagerState): TankManagerState {
    const updatedManager = startTraining(managerState);
    console.log('Started AI training');
    return updatedManager;
}

// Остановить обучение
export async function stopAITraining(managerState: TankManagerState): Promise<TankManagerState> {
    const updatedManager = await stopTraining(managerState);
    console.log('Stopped AI training');
    return updatedManager;
}

// Получение статистики обучения для отображения
export function getAITrainingStats(managerState: TankManagerState) {
    return getTrainingStats(managerState);
}

// Очистка ресурсов при выходе из игры
export async function disposeGameAI(managerState: TankManagerState): Promise<void> {
    await disposeTankManager(managerState);
}

export async function setupAISystem() {
    let currentState = await initializeGameAI();

    // Функции системы
    return {
        // Добавление AI управления ко всем танкам, которые должны его иметь
        addTanks: (tankIds: readonly number[]) => {
            for (const tankId of tankIds) {
                currentState = addControlToTank(currentState, tankId);
            }
        },

        resetTanks: () => {
            for (const tankId of currentState.agents.keys()) {
                currentState = unregisterTank(currentState, tankId);
            }
        },

        // Обновление системы на каждом тике
        update: () => {
            currentState = updateGameAI(currentState);
        },

        // Остановить обучение
        stopTraining: async () => {
            currentState = await stopAITraining(currentState);
        },

        // Получить статистику
        getStats: () => {
            return getAITrainingStats(currentState);
        },

        // Очистка ресурсов
        dispose: async () => {
            await disposeGameAI(currentState);
        },
    };
}

const aiSystem = await setupAISystem();

run();

async function run() {
    while (true) {
        try {
            await runEpisode(MAX_STEPS);
        } catch (e) {
            console.error('Error during episode:', e);
        }
    }
}

export async function runEpisode(maxSteps: number): Promise<void> {
    const { world, canvas, gameTick, destroy } = createBattlefield(TANK_COUNT_SIMULATION);

    gameTick(TICK_TIME_SIMULATION);
    aiSystem.addTanks(query(world, [Tank, TankInputTensor]));

    let steps = 0;
    return new Promise((resolve, reject) => {
        const stopInterval = macroTasks.addInterval(() => {
            if (steps >= maxSteps) {
                destroy();
                stopInterval();
                aiSystem.stopTraining();
                aiSystem.dispose();
                resolve();
            }

            try {
                steps++;
                gameTick(TICK_TIME_SIMULATION);
                aiSystem.update();
            } catch (e) {
                console.error('Error during AI update:', e);
                destroy();
                stopInterval();
                reject(e);
            }
        }, TICK_TIME_REAL);
    });
}
