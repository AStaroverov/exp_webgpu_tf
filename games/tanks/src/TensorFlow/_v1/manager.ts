import * as tf from '@tensorflow/tfjs';
import {
    createModel,
    createReplayBuffer,
    initWasmBackend,
    loadModel,
    ReplayBuffer,
    saveModel,
    trainModelOnBatch,
} from './model';
import { actAgent, createAgentState, getAgentStats, resetAgent, TankState } from './agent';
import { REWARD_WEIGHTS } from './const';

// Типы для менеджера танков
export type TankManagerState = {
    model: tf.LayersModel;
    replayBuffer: ReplayBuffer;
    agents: Map<number, TankState>;
    width: number;
    height: number;
    maxSpeed: number;
    isTraining: boolean;
    modelSaveInterval: number | null;
    frameCount: number;
    currentEpisode: number;
    metrics: TrainingMetrics;
};

export type TrainingMetrics = {
    episodeRewards: number[];
    episodeLengths: number[];
    avgReward: number;
    bestReward: number;
    totalTrainingSteps: number;
};

// Создание начального состояния менеджера
export async function createTankManager(
    width: number,
    height: number,
    maxSpeed: number,
): Promise<TankManagerState> {
    await initWasmBackend();

    let model;
    const loadedModel = await loadModel('tank-game-model');

    if (loadedModel) {
        console.log('Loaded saved model successfully');
        model = loadedModel;
    } else {
        console.log('No saved model found, using new model');
        model = createModel();
    }

    return {
        model,
        replayBuffer: createReplayBuffer(),
        agents: new Map(),
        width,
        height,
        maxSpeed,
        isTraining: false,
        modelSaveInterval: null,
        frameCount: 0,
        currentEpisode: 0,
        metrics: {
            episodeRewards: [],
            episodeLengths: [],
            avgReward: 0,
            bestReward: -Infinity,
            totalTrainingSteps: 0,
        },
    };
}

// Регистрация танка для управления моделью
export function registerTank(
    managerState: TankManagerState,
    tankEid: number,
): TankManagerState {
    // Создание нового агента для этого танка
    const agentState = createAgentState(tankEid);
    const updatedAgents = new Map(managerState.agents);
    updatedAgents.set(tankEid, agentState);

    return {
        ...managerState,
        agents: updatedAgents,
    };
}

// Удаление танка
export function unregisterTank(
    managerState: TankManagerState,
    tankEid: number,
): TankManagerState {
    const updatedAgents = new Map(managerState.agents);
    updatedAgents.delete(tankEid);

    return {
        ...managerState,
        agents: updatedAgents,
    };
}

// Сброс состояния агента
export function resetTankAgent(
    managerState: TankManagerState,
    tankEid: number,
): TankManagerState {
    const agent = managerState.agents.get(tankEid);
    if (!agent) return managerState;

    const updatedAgents = new Map(managerState.agents);
    updatedAgents.set(tankEid, resetAgent(agent));

    return {
        ...managerState,
        agents: updatedAgents,
    };
}

// Функция для обновления всех агентов
export function updateTankManager(managerState: TankManagerState): TankManagerState {
    const updatedFrameCount = managerState.frameCount + 1;

    // Собираем действия для каждого активного танка
    for (const agent of managerState.agents.values()) {
        actAgent(
            agent,
            managerState.model,
            managerState.replayBuffer,
            managerState.width,
            managerState.height,
            managerState.maxSpeed,
        );
    }

    // Периодическое обучение модели во время игры
    if (managerState.isTraining && updatedFrameCount % 60 === 0) { // каждые ~1 секунду при 60 FPS
        trainModelOnBatch(managerState.model, managerState.replayBuffer, 64);
    }

    return {
        ...managerState,
        frameCount: updatedFrameCount,
    };
}

// Запуск режима обучения
export function startTraining(
    managerState: TankManagerState,
    saveInterval: number = 5 * 60 * 1000, // 5 минут по умолчанию
): TankManagerState {
    if (managerState.isTraining) return managerState;

    console.log('Started training mode');

    // Настройка автоматического сохранения модели
    const modelSaveInterval = window.setInterval(() => {
        saveModel(managerState.model, 'tank-game-model');
    }, saveInterval);

    return {
        ...managerState,
        isTraining: true,
        modelSaveInterval,
        currentEpisode: 0,
    };
}

// Остановка режима обучения
export async function stopTraining(managerState: TankManagerState): Promise<TankManagerState> {
    if (!managerState.isTraining) return managerState;

    console.log('Stopped training mode');

    // Отключение автоматического сохранения
    if (managerState.modelSaveInterval !== null) {
        clearInterval(managerState.modelSaveInterval);
    }

    // Сохранение модели при остановке обучения
    await saveModel(managerState.model, 'tank-game-model');

    return {
        ...managerState,
        isTraining: false,
        modelSaveInterval: null,
    };
}

// Обновление настроек наград
export function updateRewardWeights(
    newWeights: Partial<typeof REWARD_WEIGHTS>,
): void {
    Object.assign(REWARD_WEIGHTS, newWeights);
    console.log('Updated reward weights:', REWARD_WEIGHTS);
}

// Получение статистики обучения
export function getTrainingStats(managerState: TankManagerState) {
    // Сбор метрик от всех агентов
    const agentStats = Array.from(managerState.agents.values()).map(agent => getAgentStats(agent));

    // Объединение метрик
    const combinedEpisodeRewards: number[] = [];
    const bulletStats = { total: 0, avoided: 0 };
    const combinedRewardHistory: { [key: string]: number[] } = {};

    for (const stats of agentStats) {
        combinedEpisodeRewards.push(...stats.episodeRewards);
        bulletStats.total += stats.bulletMemory.count;
        bulletStats.avoided += stats.bulletMemory.avoided;

        // Объединение истории наград по компонентам
        for (const [key, values] of Object.entries(stats.rewardHistory)) {
            if (!combinedRewardHistory[key]) {
                combinedRewardHistory[key] = [];
            }
            combinedRewardHistory[key].push(...values);
        }
    }

    // Расчет средних значений по компонентам
    const avgComponentRewards: { [key: string]: number } = {};
    for (const [key, values] of Object.entries(combinedRewardHistory)) {
        if (values.length > 0) {
            avgComponentRewards[key] = values.reduce((a, b) => a + b, 0) / values.length;
        } else {
            avgComponentRewards[key] = 0;
        }
    }

    // Обновление общих метрик
    let updatedMetrics = { ...managerState.metrics };
    if (combinedEpisodeRewards.length > 0) {
        const sum = combinedEpisodeRewards.reduce((a, b) => a + b, 0);
        updatedMetrics.avgReward = sum / combinedEpisodeRewards.length;
        updatedMetrics.bestReward = Math.max(updatedMetrics.bestReward, ...combinedEpisodeRewards);
        updatedMetrics.totalTrainingSteps += combinedEpisodeRewards.length;
        updatedMetrics.episodeRewards.push(...combinedEpisodeRewards);
    }

    return {
        ...updatedMetrics,
        currentEpisode: managerState.currentEpisode,
        activeAgents: managerState.agents.size,
        isTraining: managerState.isTraining,
        rewardComponents: avgComponentRewards,
        bulletStats: {
            total: bulletStats.total,
            avoided: bulletStats.avoided,
            avoidanceRate: bulletStats.total > 0 ? bulletStats.avoided / bulletStats.total : 0,
        },
    };
}

// Очистка ресурсов
export async function disposeTankManager(managerState: TankManagerState): Promise<void> {
    if (managerState.isTraining) {
        await stopTraining(managerState);
    }

    if (managerState.model) {
        managerState.model.dispose();
    }

    managerState.agents.clear();
}