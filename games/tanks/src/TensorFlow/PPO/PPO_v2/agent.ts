import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM, INPUT_DIM } from '../../Common/consts.ts';
import { Config, getCurrentConfig } from '../Common/config.ts';
import { Memory } from '../Common/Memory.ts';
import { abs, floor } from '../../../../../../lib/math.ts';
import { isDevtoolsOpen } from '../../Common/utils.ts';
import { computeLogProbTanh } from '../../Common/computeLogProb.ts';
import { createPolicyNetwork, createValueNetwork } from '../../Common/models.ts';
import { trainPolicyNetwork, trainValueNetwork } from '../Common/train.ts';

// Общий PPO агент для всех танков
export class SharedTankPPOAgent {
    private iteration = 0;
    private memory: Memory;
    private valueNetwork: tf.LayersModel;   // Сеть критика
    private policyNetwork: tf.LayersModel;  // Сеть политики
    private policyOptimizer!: tf.Optimizer;        // Оптимизатор для policy network
    private valueOptimizer!: tf.Optimizer;         // Оптимизатор для value network
    private config!: Config;

    private logger: {
        episodeRewards: number[];
        episodeLengths: number[];
        losses: {
            policy: number[];
            value: number[];
        };
    };

    constructor() {
        // Инициализируем с значениями из конфига
        this.memory = new Memory();
        this.applyConfig(getCurrentConfig());

        // Инициализируем логгер
        this.logger = {
            episodeRewards: [],
            episodeLengths: [],
            losses: {
                policy: [],
                value: [],
            },
        };

        // Создаем модели
        this.policyNetwork = createPolicyNetwork(this.config.hiddenLayers);
        this.valueNetwork = createValueNetwork(this.config.hiddenLayers);

        console.log(`Shared PPO Agent initialized with experiment: ${ this.config.name }`);
    }

    // Освобождение ресурсов
    dispose() {
        this.policyNetwork.dispose();
        this.valueNetwork.dispose();
        this.memoryDispose();
        console.log('PPO Agent resources disposed');
    }

    memoryDispose() {
        this.memory.dispose();
    }

    // Методы для сохранения опыта в буфер
    rememberAction(tankId: number, state: Float32Array, action: Float32Array, logProb: number, value: number) {
        this.memory.addFirstPart(tankId, state, action, logProb, value);
    }

    rememberReward(tankId: number, reward: number, done: boolean, isLast = false) {
        this.memory.updateSecondPart(tankId, reward, done, isLast);
    }

    tryTrain(useTail: boolean): boolean {
        const batchSize = this.config.batchSize;
        const memorySize = this.memory.size();

        if (useTail && memorySize < batchSize / 6) {
            return false;
        }
        if (!useTail && memorySize < batchSize) {
            return false;
        }

        const batch = this.memory.getBatch(
            this.config.gamma,
            this.config.lam,
        );
        const epochs = useTail && batch.size < batchSize
            ? floor(batch.size / batchSize * this.config.epochs)
            : this.config.epochs;

        if (epochs === 0) {
            return false;
        }

        let policyLossSum = 0, valueLossSum = 0;

        console.log(`[Train]: Iteration ${ this.iteration++ }, Batch size: ${ batch.size }, Epochs: ${ epochs }`);

        const prevWeights = isDevtoolsOpen() ? this.policyNetwork.getWeights().map(w => w.dataSync()) as Float32Array[] : null;

        const tStates = tf.tensor(batch.states, [batch.size, INPUT_DIM]);
        const tActions = tf.tensor(batch.actions, [batch.size, ACTION_DIM]);
        const tLogProbs = tf.tensor(batch.logProbs, [batch.size]);
        const tValues = tf.tensor(batch.values, [batch.size]);
        const tAdvantages = tf.tensor(batch.advantages, [batch.size]);
        const tReturns = tf.tensor(batch.returns, [batch.size]);

        for (let i = 0; i < epochs; i++) {
            // Обучение политики
            const policyLoss = trainPolicyNetwork(
                this.policyNetwork, this.policyOptimizer, this.config,
                tStates, tActions, tLogProbs, tAdvantages,
            );
            policyLossSum += policyLoss;

            // Обучение критика
            const valueLoss = trainValueNetwork(
                this.valueNetwork, this.valueOptimizer, this.config,
                tStates, tReturns, tValues,
            );
            valueLossSum += valueLoss;

            console.log(`[Train]: Epoch: ${ i }, Policy loss: ${ policyLoss.toFixed(4) }, Value loss: ${ valueLoss.toFixed(4) }`);
        }

        tStates.dispose();
        tActions.dispose();
        tLogProbs.dispose();
        tValues.dispose();
        tAdvantages.dispose();
        tReturns.dispose();

        const newWeights = isDevtoolsOpen() ? this.policyNetwork.getWeights().map(w => w.dataSync()) as Float32Array[] : null;

        isDevtoolsOpen() && console.log('>> WEIGHTS SUM ABS DELTA', newWeights!.reduce((acc, w, i) => {
            return acc + abs(w.reduce((a, b, j) => a + abs(b - prevWeights![i][j]), 0));
        }, 0));

        this.memory.dispose();

        const avgPolicyLoss = policyLossSum / epochs;
        const avgValueLoss = valueLossSum / epochs;

        this.logger.losses.policy.push(avgPolicyLoss);
        this.logger.losses.value.push(avgValueLoss);

        return true;
    }

    // Логирование данных эпизода
    logEpisode(reward: number, length: number) {
        this.logger.episodeRewards.push(reward);
        this.logger.episodeLengths.push(length);
    }

    // Получение текущей статистики для отображения
    getStats() {
        const last10Rewards = this.logger.episodeRewards.slice(-10);
        const avgReward = last10Rewards.length > 0
            ? last10Rewards.reduce((a, b) => a + b, 0) / last10Rewards.length
            : 0;

        const last10PolicyLoss = this.logger.losses.policy.slice(-10);
        const avgPolicyLoss = last10PolicyLoss.length > 0
            ? last10PolicyLoss.reduce((a, b) => a + b, 0) / last10PolicyLoss.length
            : 0;

        const last10ValueLoss = this.logger.losses.value.slice(-10);
        const avgValueLoss = last10ValueLoss.length > 0
            ? last10ValueLoss.reduce((a, b) => a + b, 0) / last10ValueLoss.length
            : 0;

        return {
            memorySize: this.memory.size(),
            avgReward: avgReward,
            lastReward: this.logger.episodeRewards[this.logger.episodeRewards.length - 1] ?? 0,
            avgPolicyLoss: avgPolicyLoss,
            avgValueLoss: avgValueLoss,
            experimentName: this.config.name,
        };
    }

    // Сохранение модели
    async save() {
        try {
            await this.policyNetwork.save('indexeddb://tank-rl-policy-model');
            await this.valueNetwork.save('indexeddb://tank-rl-value-model');

            localStorage.setItem('tank-rl-agent-state', JSON.stringify({
                iteration: this.iteration,
                config: this.config,
                timestamp: new Date().toISOString(),
                rewards: this.logger.episodeRewards.slice(-100),
                losses: {
                    policy: this.logger.losses.policy.slice(-100),
                    value: this.logger.losses.value.slice(-100),
                },
            }));

            console.log(`PPO models saved`);
            return true;
        } catch (error) {
            console.error('Error saving PPO models:', error);
            return false;
        }
    }

    async download() {
        return this.policyNetwork.save('downloads://tank-rl-policy-model');
    }

    // Загрузка модели из хранилища
    async load() {
        try {
            this.policyNetwork = await tf.loadLayersModel('indexeddb://tank-rl-policy-model');
            this.valueNetwork = await tf.loadLayersModel('indexeddb://tank-rl-value-model');

            console.log('PPO models loaded successfully');

            const metadata = JSON.parse(localStorage.getItem('tank-rl-agent-state') ?? '{}');
            if (metadata) {
                this.iteration = metadata.iteration || 0;

                if (metadata.config) {
                    this.applyConfig(metadata.config);
                }

                // Загружаем историю потерь и наград, если она есть
                if (metadata.losses) {
                    this.logger.losses = metadata.losses;
                }

                if (metadata.rewards && metadata.rewards.length > 0) {
                    this.logger.episodeRewards = metadata.rewards;
                }

                // Проверяем, не изменился ли эксперимент
                if (metadata.experimentName && metadata.experimentName !== this.config.name) {
                    console.warn(`Loaded model was trained with experiment "${ metadata.experimentName }", but current experiment is "${ this.config.name }"`);
                }
            }
            return true;
        } catch (error) {
            console.warn('Could not load PPO models, starting with new ones:', error);
            return false;
        }
    }

    act(state: Float32Array): {
        rawActions: Float32Array,
        actions: Float32Array,
        logProb: number,
        value: number
    } {
        return tf.tidy(() => {
            const stateTensor = tf.tensor1d(state).expandDims(0);
            const predict = this.policyNetwork.predict(stateTensor) as tf.Tensor;
            const rawOutputSqueezed = predict.squeeze(); // [ACTION_DIM * 2] при batch=1
            const outMean = rawOutputSqueezed.slice([0], [ACTION_DIM]);   // ACTION_DIM штук
            const outLogStd = rawOutputSqueezed.slice([ACTION_DIM], [ACTION_DIM]);
            const clippedLogStd = outLogStd.clipByValue(-2, 0.2);
            const std = clippedLogStd.exp();
            const noise = tf.randomNormal([ACTION_DIM]).mul(std);
            const action = outMean.add(noise);
            const logProb = computeLogProbTanh(action, outMean, std);
            const value = this.valueNetwork.predict(stateTensor) as tf.Tensor;

            return {
                rawActions: action.dataSync() as Float32Array,
                actions: action.tanh().dataSync() as Float32Array,
                logProb: logProb.dataSync()[0],
                value: value.squeeze().dataSync()[0],
            };
        });
    }

    private applyConfig(config: Config) {
        this.config = config;
        // TODO: useless on load, fix it later
        this.policyOptimizer = tf.train.adam(this.config.learningRatePolicy);
        this.valueOptimizer = tf.train.adam(this.config.learningRateValue);
    }
}

// Создаем синглтон агента
let sharedAgent: SharedTankPPOAgent | null = null;

// Получение или создание общего агента
export function getSharedAgent(): SharedTankPPOAgent {
    if (!sharedAgent) {
        sharedAgent = new SharedTankPPOAgent();
    }
    return sharedAgent;
}

// Очистка общего агента
export function disposeSharedAgent() {
    if (sharedAgent) {
        sharedAgent.dispose();
        sharedAgent = null;
    }
}
