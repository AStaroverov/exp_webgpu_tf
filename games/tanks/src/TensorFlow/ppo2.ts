import * as tf from '@tensorflow/tfjs';
import { layers, LayersModel, Scalar, sequential } from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../ECS/Components/TankState';
import { hypot, max, smoothstep } from '../../../../lib/math';
import { clamp, inRange } from 'lodash-es';
import { TankController } from '../ECS/Components/TankController.ts';
import { createBattlefield } from './createBattlefield.ts';
import { query } from 'bitecs';
import { Tank } from '../ECS/Components/Tank.ts';
import { macroTasks } from '../../../../lib/TasksScheduler/macroTasks.ts';

setWasmPaths('/node_modules/@tensorflow/tfjs-backend-wasm/dist/');
await tf.setBackend('wasm');

// Configuration constants
export const TANK_COUNT_SIMULATION = 6; // Reduced to make training more manageable
const TICK_TIME_REAL = 1;
const TICK_TIME_SIMULATION = 16.6667 * 2;
const INPUT_DIM = 63; // Tank state dimensions (same as your original implementation)
const ACTION_DIM = 5; // [shoot, move, turn, targetX, targetY]

const PPO_EPOCHS = 4;
const BATCH_SIZE = 64;
const CLIP_EPSILON = 0.2;
// Индивидуальные веса энтропии для разных действий
const ENTROPY_WEIGHTS = [0.02, 0.01, 0.01, 0.005, 0.005]; // [shoot, move, turn, targetX, targetY]
const GAMMA = 0.99; // Discount factor
const LAMBDA = 0.95; // GAE parameter

const TANK_RADIUS = 80;

// Dynamic learning rate based on episode count
function getLearningRate(episodeNum: number, initialLR: number = 0.0003): number {
    return initialLR * Math.max(0.1, 1.0 - episodeNum / 1000);
}

// Prioritized Experience Buffer for better experience sampling
class PrioritizedExperienceBuffer {
    states: tf.Tensor[] = [];
    actions: tf.Tensor[] = [];
    oldLogProbs: tf.Tensor[] = [];
    rewards: number[] = [];
    values: tf.Tensor[] = [];
    dones: boolean[] = []; // Добавлено для отметки терминальных состояний
    priorities: number[] = []; // Приоритеты для выборки
    alpha: number = 0.6; // Показатель степени для приоритета (контролирует степень приоритизации)
    beta: number = 0.4; // Исходное значение бета для важности выборки (начинаем с 0.4 и увеличиваем до 1)
    betaAnnealing: number = 0.001; // Скорость увеличения бета

    constructor(private capacity: number = 1024) {
    }

    get size() {
        return this.states.length;
    }

    add(state: tf.Tensor, action: tf.Tensor, logProb: tf.Tensor, reward: number, value: tf.Tensor, done: boolean, priority: number = 1.0) {
        this.states.push(state);
        this.actions.push(action);
        this.oldLogProbs.push(logProb);
        this.rewards.push(reward);
        this.values.push(value);
        this.dones.push(done);
        this.priorities.push(priority);

        // Trim if exceeding capacity
        if (this.states.length > this.capacity) {
            // Clean up tensors before removing
            this.states[0].dispose();
            this.actions[0].dispose();
            this.oldLogProbs[0].dispose();
            this.values[0].dispose();

            this.states.shift();
            this.actions.shift();
            this.oldLogProbs.shift();
            this.rewards.shift();
            this.values.shift();
            this.dones.shift();
            this.priorities.shift();
        }
    }

    clear() {
        // Clean up tensors
        this.states.forEach(t => t.dispose());
        this.actions.forEach(t => t.dispose());
        this.oldLogProbs.forEach(t => t.dispose());
        this.values.forEach(t => t.dispose());

        this.states = [];
        this.actions = [];
        this.oldLogProbs = [];
        this.rewards = [];
        this.values = [];
        this.dones = [];
        this.priorities = [];
    }

    // Update priorities for experiences (typically after computing TD errors)
    updatePriorities(indices: number[], newPriorities: number[]) {
        for (let i = 0; i < indices.length; i++) {
            if (indices[i] < this.priorities.length) {
                this.priorities[indices[i]] = Math.max(0.00001, newPriorities[i]);
            }
        }
    }

    // Calculate advantages using Generalized Advantage Estimation (GAE)
    // Улучшенная версия с учетом флагов терминальных состояний
    computeReturnsAndAdvantages(): [tf.Tensor, tf.Tensor] {
        const returns: number[] = new Array(this.rewards.length);
        const advantages: number[] = new Array(this.rewards.length);

        let nextReturn = 0; // Для терминального состояния, начинаем с нуля
        let nextAdvantage = 0;

        for (let i = this.rewards.length - 1; i >= 0; i--) {
            const reward = this.rewards[i];
            const value = this.values[i].dataSync()[0];
            const done = this.dones[i];

            // Для терминальных состояний, следующее возвращаемое значение - это только reward
            const nextStateValue = done ? 0 : (i < this.rewards.length - 1 ? this.values[i + 1].dataSync()[0] : 0);

            // Compute TD error and GAE
            const delta = reward + (done ? 0 : GAMMA * nextStateValue) - value;
            nextAdvantage = delta + (done ? 0 : GAMMA * LAMBDA * nextAdvantage);
            nextReturn = reward + (done ? 0 : GAMMA * nextReturn);

            returns[i] = nextReturn;
            advantages[i] = nextAdvantage;
        }

        return [
            tf.tensor1d(returns),
            tf.tensor1d(advantages),
        ];
    }

    // Выборка с приоритезацией
    getBatch(batchSize: number): [tf.Tensor[], tf.Tensor[], tf.Tensor[], tf.Tensor, tf.Tensor, number[]] {
        if (this.size < batchSize) {
            throw new Error(`Buffer size (${ this.size }) smaller than requested batch size (${ batchSize })`);
        }

        // Анализ бета для корректировки важности выборки
        this.beta = Math.min(1.0, this.beta + this.betaAnnealing);

        // Перевести приоритеты в вероятности выборки
        const prioritySum = this.priorities.reduce((a, b) => a + Math.pow(b, this.alpha), 0);
        const probabilities = this.priorities.map(p => Math.pow(p, this.alpha) / prioritySum);

        // Выборка индексов на основе вероятностей
        const indices: number[] = [];
        for (let i = 0; i < batchSize; i++) {
            let idx = 0;
            let r = Math.random();
            let sum = 0;
            for (let j = 0; j < probabilities.length; j++) {
                sum += probabilities[j];
                if (r <= sum) {
                    idx = j;
                    break;
                }
            }
            indices.push(idx);
        }

        // Рассчитать веса важности выборки для корректировки смещения
        const maxWeight = Math.pow(this.size * Math.min(...probabilities), -this.beta);
        const weights = indices.map(i => Math.pow(this.size * probabilities[i], -this.beta) / maxWeight);

        // Get batch elements
        const batchStates = indices.map(i => this.states[i]);
        const batchActions = indices.map(i => this.actions[i]);
        const batchLogProbs = indices.map(i => this.oldLogProbs[i]);

        // Calculate returns and advantages
        const [returns, advantages] = this.computeReturnsAndAdvantages();

        const batchReturns = tf.gather(returns, indices);
        const batchAdvantages = tf.gather(advantages, indices);

        // Normalize advantages
        const mean = tf.mean(batchAdvantages);
        const std = tf.sqrt(tf.mean(tf.square(tf.sub(batchAdvantages, mean))));
        const normalizedAdvantages = tf.div(tf.sub(batchAdvantages, mean), tf.add(std, 1e-8));

        // Умножить преимущества на веса важности
        const weightedAdvantages = tf.mul(normalizedAdvantages, tf.tensor1d(weights));

        returns.dispose();
        advantages.dispose();
        mean.dispose();
        std.dispose();
        normalizedAdvantages.dispose();

        return [batchStates, batchActions, batchLogProbs, batchReturns, weightedAdvantages, indices];
    }
}

// Input statistics for normalization
class InputNormalizer {
    private mean: number[];
    private var: number[];
    private count: number;
    private runningStats: boolean;

    constructor(inputDim: number, runningStats: boolean = true) {
        this.mean = new Array(inputDim).fill(0);
        this.var = new Array(inputDim).fill(1);
        this.count = 0;
        this.runningStats = runningStats;
    }

    update(inputVector: Float32Array) {
        if (!this.runningStats) return;

        this.count++;
        for (let i = 0; i < inputVector.length; i++) {
            const delta = inputVector[i] - this.mean[i];
            this.mean[i] += delta / this.count;
            this.var[i] += delta * (inputVector[i] - this.mean[i]);
        }
    }

    normalize(inputVector: Float32Array): Float32Array {
        const normalizedInput = new Float32Array(inputVector.length);

        if (this.count > 1) {
            for (let i = 0; i < inputVector.length; i++) {
                const std = Math.sqrt(this.var[i] / (this.count - 1));
                normalizedInput[i] = (inputVector[i] - this.mean[i]) / (std + 1e-8);
            }
        } else {
            // Если мало данных, просто копируем вектор
            normalizedInput.set(inputVector);
        }

        return normalizedInput;
    }

    // Сохранение и загрузка статистики
    async save(key: string = 'tank-input-stats') {
        const saveObj = {
            mean: this.mean,
            var: this.var,
            count: this.count,
        };
        localStorage.setItem(key, JSON.stringify(saveObj));
    }

    async load(key: string = 'tank-input-stats'): Promise<boolean> {
        try {
            const saved = localStorage.getItem(key);
            if (saved) {
                const stats = JSON.parse(saved);
                this.mean = stats.mean;
                this.var = stats.var;
                this.count = stats.count;
                return true;
            }
            return false;
        } catch (error) {
            console.error('Failed to load input stats:', error);
            return false;
        }
    }
}

// Create Actor (Policy) Network with LSTM layer for better handling of partial observations
function createActorModel(): { meanModel: LayersModel, stdModel: LayersModel } {
    // Mean model with LSTM
    const meanModel = sequential();
    meanModel.add(layers.dense({
        inputShape: [INPUT_DIM],
        units: 128,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
    }));

    // Reshape for LSTM
    meanModel.add(layers.reshape({ targetShape: [1, 128] }));

    // LSTM layer for sequence modeling
    meanModel.add(layers.lstm({
        units: 64,
        returnSequences: false,
        kernelInitializer: 'glorotNormal',
    }));

    meanModel.add(layers.dense({
        units: ACTION_DIM,
        activation: 'tanh',  // Using tanh for [-1, 1] range
        kernelInitializer: 'glorotNormal',
    }));

    // Std model (log standard deviations)
    const stdModel = sequential();
    stdModel.add(layers.dense({
        inputShape: [INPUT_DIM],
        units: 64,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
    }));
    stdModel.add(layers.dense({
        units: ACTION_DIM,
        // Используем softplus вместо tanh для более мягкого ограничения логарифма стандартного отклонения
        activation: 'softplus',
        biasInitializer: tf.initializers.constant({ value: -1.0 }), // Начинаем с меньшего стандартного отклонения
    }));

    // Compile the models individually
    meanModel.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    stdModel.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    return { meanModel, stdModel };
}

// Create Critic (Value) Network - также с LSTM
function createCriticModel(): LayersModel {
    const model = sequential();

    // Input layer
    model.add(layers.dense({
        inputShape: [INPUT_DIM],
        units: 128,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
    }));

    // Reshape for LSTM
    model.add(layers.reshape({ targetShape: [1, 128] }));

    // LSTM layer for sequence modeling
    model.add(layers.lstm({
        units: 64,
        returnSequences: false,
        kernelInitializer: 'glorotNormal',
    }));

    // Output layer - single value estimate
    model.add(layers.dense({
        units: 1,
        kernelInitializer: 'glorotNormal',
    }));

    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError',
    });

    return model;
}

// Implementation of improved PPO Agent
class PPOAgent {
    actorMean: LayersModel;
    actorStd: LayersModel;
    critic: LayersModel;
    buffer: PrioritizedExperienceBuffer;
    inputNormalizer: InputNormalizer;
    episodeCount: number;

    constructor() {
        const { meanModel, stdModel } = createActorModel();
        this.actorMean = meanModel;
        this.actorStd = stdModel;
        this.critic = createCriticModel();
        this.buffer = new PrioritizedExperienceBuffer();
        this.inputNormalizer = new InputNormalizer(INPUT_DIM);
        this.episodeCount = 0;
    }

    // Sample action from policy
    sampleAction(state: tf.Tensor): [tf.Tensor, tf.Tensor] {
        return tf.tidy(() => {
            // Get mean and log std from actor networks
            const meanTensor = this.actorMean.predict(state) as tf.Tensor;
            const stdTensor = this.actorStd.predict(state) as tf.Tensor;

            // Sample from normal distribution
            const epsilonTensor = tf.randomNormal(meanTensor.shape);
            const actionTensor = tf.add(meanTensor, tf.mul(stdTensor, epsilonTensor));

            // Constrain actions to valid range [-1, 1] if needed
            const clippedActionTensor = tf.clipByValue(actionTensor, -1, 1);

            // Calculate log probability of the action
            const logProbTensor = this.calculateLogProb(meanTensor, stdTensor, clippedActionTensor);

            return [clippedActionTensor, logProbTensor];
        });
    }

    // Calculate log probability of an action with improved numerical stability
    calculateLogProb(mean: tf.Tensor, std: tf.Tensor, action: tf.Tensor): tf.Tensor {
        return tf.tidy(() => {
            // Formula: -0.5 * (((action - mean) / std)^2 + 2 * log(std) + log(2*pi))
            const diff = tf.sub(action, mean);
            const scaled = tf.div(diff, tf.add(std, 1e-8)); // Избегаем деления на ноль
            const squaredScaled = tf.square(scaled);

            // Используем log1p для большей численной стабильности
            const logVar = tf.log(tf.add(tf.square(std), 1e-8));

            const logProb = tf.mul(
                tf.scalar(-0.5),
                tf.add(
                    tf.add(squaredScaled, tf.mul(tf.scalar(2), logVar)),
                    tf.scalar(Math.log(2 * Math.PI)),
                ),
            );

            // Sum across action dimensions
            return tf.sum(logProb, -1);
        });
    }

    // Evaluate state with critic
    evaluateState(state: tf.Tensor): tf.Tensor {
        return tf.tidy(() => {
            return this.critic.predict(state) as tf.Tensor;
        });
    }

    // Store experience in buffer with priority
    storeExperience(state: tf.Tensor, action: tf.Tensor, logProb: tf.Tensor, reward: number, value: tf.Tensor, done: boolean = false) {
        // Использовать абсолютную величину вознаграждения как начальный приоритет
        const priority = Math.abs(reward) + 0.01; // Добавляем малую константу, чтобы избежать нулевых приоритетов
        this.buffer.add(state.clone(), action.clone(), logProb.clone(), reward, value.clone(), done, priority);
    }

    // Train using improved PPO algorithm
    async train() {
        if (this.buffer.size < BATCH_SIZE) {
            console.log('Not enough samples for training');
            return {};
        }

        console.log(`Training PPO with ${ this.buffer.size } samples, episode count: ${ this.episodeCount }`);

        // Динамически определяем скорость обучения на основе номера эпизода
        const learningRate = getLearningRate(this.episodeCount);
        console.log(`Current learning rate: ${ learningRate.toFixed(6) }`);

        // Tracking for priorities update
        const indices: number[] = [];
        const newPriorities: number[] = [];
        let totalActorLoss = 0;
        let totalCriticLoss = 0;

        for (let epoch = 0; epoch < PPO_EPOCHS; epoch++) {
            // Get training batch with prioritized experience replay
            const [
                batchStates,
                batchActions,
                batchOldLogProbs,
                batchReturns,
                batchAdvantages,
                batchIndices,
            ] = this.buffer.getBatch(BATCH_SIZE);

            // Сохраним индексы для обновления приоритетов
            indices.push(...batchIndices);

            // Optimize actor
            const actorLoss = await this.optimizeActor(
                batchStates,
                batchActions,
                batchOldLogProbs,
                batchAdvantages,
                learningRate,
            );

            // Optimize critic
            const criticLoss = await this.optimizeCritic(
                batchStates,
                batchReturns,
                learningRate,
            );

            totalActorLoss += actorLoss;
            totalCriticLoss += criticLoss;

            console.log(`Epoch ${ epoch + 1 }/${ PPO_EPOCHS }: Actor Loss = ${ actorLoss.toFixed(4) }, Critic Loss = ${ criticLoss.toFixed(4) }`);

            // Обновим приоритеты на основе ошибок обучения
            // Используем абсолютное значение ошибки критика как новый приоритет
            const criticPreds = tf.tidy(() => {
                // Объединяем состояния в партию
                const stateBatch = tf.concat(batchStates.map(s => s.reshape([1, -1])));
                // Получаем предсказания критика
                const predictions = this.critic.predict(stateBatch) as tf.Tensor;
                // Вычисляем TD-ошибки
                const tdErrors = tf.abs(tf.sub(batchReturns, predictions));
                return tdErrors.dataSync();
            });

            // Добавляем новые приоритеты
            for (let i = 0; i < batchIndices.length; i++) {
                newPriorities.push(criticPreds[i] + 0.01); // Добавляем небольшое значение, чтобы избежать нулевых приоритетов
            }

            // Dispose tensors
            batchReturns.dispose();
            batchAdvantages.dispose();
        }

        // Update priorities in the buffer
        this.buffer.updatePriorities(indices, newPriorities);

        // Clear buffer after training (for on-policy algorithm)
        // this.buffer.clear(); // Закомментировано для поддержки смешанного режима on-policy/off-policy

        this.episodeCount++;

        return {
            actorLoss: totalActorLoss / PPO_EPOCHS,
            criticLoss: totalCriticLoss / PPO_EPOCHS,
        };
    }

    // Optimize actor using PPO clip objective with improved memory management
    async optimizeActor(
        states: tf.Tensor[],
        actions: tf.Tensor[],
        oldLogProbs: tf.Tensor[],
        advantages: tf.Tensor,
        learningRate: number,
    ): Promise<number> {
        // Combine states into a batch
        const stateBatch = tf.concat(states.map(s => s.reshape([1, -1])));
        const actionBatch = tf.concat(actions.map(a => a.reshape([1, -1])));
        const oldLogProbBatch = tf.concat(oldLogProbs.map(lp => lp.reshape([1, -1])));

        // Create optimizer with dynamic learning rate
        const meanOptimizer = tf.train.adam(learningRate);
        const stdOptimizer = tf.train.adam(learningRate);

        // Custom training step with gradients
        const { value, grads } = tf.variableGrads(() => {
            // TF.js tidy to manage memory
            return tf.tidy(() => {
                // Forward pass to get current policy distribution
                const meanTensor = this.actorMean.predict(stateBatch) as tf.Tensor;
                const stdTensor = this.actorStd.predict(stateBatch) as tf.Tensor;

                // Calculate new log probabilities
                const newLogProbs = this.calculateLogProb(meanTensor, stdTensor, actionBatch);

                // Calculate ratio and clipped ratio for PPO
                const ratio = tf.exp(tf.sub(newLogProbs, oldLogProbBatch.reshape([-1])));
                const clippedRatio = tf.clipByValue(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON);

                // PPO loss terms
                const surrogateLoss1 = tf.mul(ratio, advantages);
                const surrogateLoss2 = tf.mul(clippedRatio, advantages);
                const ppoLoss = tf.neg(tf.mean(tf.minimum(surrogateLoss1, surrogateLoss2)));

                // Entropy bonus for exploration - с различными весами для разных действий
                // Рассчитываем энтропию для каждого измерения
                const entropyPerDimension = tf.tidy(() => {
                    const entropyValues = [];
                    for (let i = 0; i < ACTION_DIM; i++) {
                        const entropyDim = tf.mean(tf.add(
                            tf.slice(stdTensor, [0, i], [-1, 1]),
                            tf.scalar(0.5 * Math.log(2 * Math.PI * Math.E)),
                        ));
                        entropyValues.push(entropyDim);
                    }
                    return entropyValues;
                });

                // Применяем разные веса к разным измерениям
                const weightedEntropy = tf.tidy(() => {
                    let totalEntropy = tf.scalar(0);
                    for (let i = 0; i < ACTION_DIM; i++) {
                        totalEntropy = tf.add(
                            totalEntropy,
                            tf.mul(entropyPerDimension[i], tf.scalar(ENTROPY_WEIGHTS[i])),
                        );
                    }
                    return totalEntropy;
                });

                // Dispose entropy tensors
                entropyPerDimension.forEach(t => t.dispose());

                const totalLoss = tf.sub(ppoLoss, weightedEntropy);

                return totalLoss;
            }) as Scalar;
        });

        // Apply optimization
        try {
            // Apply gradients to mean model
            meanOptimizer.applyGradients(
                Object.entries(grads)
                    .filter(([name]) => name.includes('dense'))
                    .map(([name, tensor]) => ({ name, tensor })),
            );

            // Apply gradients to std model
            stdOptimizer.applyGradients(
                Object.entries(grads)
                    .filter(([name]) => name.includes('dense'))
                    .map(([name, tensor]) => ({ name, tensor })),
            );

            const lossValue = value.dataSync()[0];

            // Cleanup
            stateBatch.dispose();
            actionBatch.dispose();
            oldLogProbBatch.dispose();
            Object.values(grads).forEach(g => g.dispose());
            value.dispose();

            return lossValue;
        } catch (error) {
            console.error('Error applying gradients:', error);

            // Cleanup on error
            stateBatch.dispose();
            actionBatch.dispose();
            oldLogProbBatch.dispose();
            Object.values(grads).forEach(g => g.dispose());
            value.dispose();

            return 0;
        }
    }

    // Optimize critic using MSE loss with improved memory management
    async optimizeCritic(
        states: tf.Tensor[],
        returns: tf.Tensor,
        learningRate: number,
    ): Promise<number> {
        // Combine states into a batch
        const stateBatch = tf.concat(states.map(s => s.reshape([1, -1])));

        // Создаем новый оптимизатор с нужной скоростью обучения
        const criticOptimizer = tf.train.adam(learningRate);

        // Перекомпилируем модель с новым оптимизатором
        this.critic.compile({
            optimizer: criticOptimizer,
            loss: 'meanSquaredError',
        });

        // Optimize critic using built-in fit method with custom optimizer
        const history = await this.critic.fit(stateBatch, returns, {
            epochs: 1,
            batchSize: states.length,
            verbose: 0,
            // optimizer: criticOptimizer,
        });

        // Cleanup
        stateBatch.dispose();

        return history.history.loss[0] as number;
    }

    async saveModels(version: string = 'latest') {
        const suffix = version === 'latest' ? '' : `-${ version }`;

        await this.actorMean.save(`indexeddb://tank-actor-mean-model${ suffix }`);
        await this.actorStd.save(`indexeddb://tank-actor-std-model${ suffix }`);
        await this.critic.save(`indexeddb://tank-critic-model${ suffix }`);
        await this.inputNormalizer.save(`tank-input-stats${ suffix }`);

        // Сохраним также счетчик эпизодов
        localStorage.setItem(`tank-episode-count${ suffix }`, this.episodeCount.toString());

        console.log(`Models and input statistics saved successfully (version: ${ version })`);
    }

    async loadModels(version: string = 'latest') {
        try {
            const suffix = version === 'latest' ? '' : `-${ version }`;

            this.actorMean = await tf.loadLayersModel(`indexeddb://tank-actor-mean-model${ suffix }`);
            this.actorStd = await tf.loadLayersModel(`indexeddb://tank-actor-std-model${ suffix }`);
            this.critic = await tf.loadLayersModel(`indexeddb://tank-critic-model${ suffix }`);
            this.actorMean.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
            this.actorStd.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
            this.critic.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

            // Загрузим статистику нормализации
            await this.inputNormalizer.load(`tank-input-stats${ suffix }`);

            // Загрузим счетчик эпизодов
            const episodeCount = localStorage.getItem(`tank-episode-count${ suffix }`);
            if (episodeCount) {
                this.episodeCount = parseInt(episodeCount);
            }

            console.log(`Models and input statistics loaded successfully (version: ${ version })`);
            return true;
        } catch (error) {
            console.log(`Failed to load models (version: ${ version }):`, error);
            return false;
        }
    }
}

// Enhanced reward calculation
function calculateReward(
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

            // Дополнительное вознаграждение за прицеливание в ближайшего врага
            if (distFromTankToEnemy === closestEnemyDist && aimQuality > 0.7) {
                rewardRecord.aim += 0.3;
            }

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
        const bulletDirectionX = bulletVx / bulletSpeed;
        const bulletDirectionY = bulletVy / bulletSpeed;

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

    return {
        totalReward,
        rewards: rewardRecord,
    };
}

// Function to run a simulation episode with improved handling and rewards
async function runEpisode(agent: PPOAgent, maxSteps: number): Promise<number> {
    const episodeReward: number[] = [];
    const episodeRewardComponents: Record<string, number[]> = {
        map: [],
        aim: [],
        avoidBullets: [],
        avoidEnemies: [],
        health: [],
        damageDealt: [],
        movement: [],
        survival: [],
    };

    return new Promise(resolve => {
        const { world, canvas, gameTick, destroy } = createBattlefield(TANK_COUNT_SIMULATION);

        // Maps to track tank state and metrics
        const mapTankToState = new Map<number, tf.Tensor>();
        const mapTankToAction = new Map<number, tf.Tensor>();
        const mapTankToLogProb = new Map<number, tf.Tensor>();
        const mapTankToValue = new Map<number, tf.Tensor>();
        const mapTankToReward = new Map<number, number>();
        const mapTankToHealth = new Map<number, number>(); // Track previous health for damage calculation
        const mapTankToDone = new Map<number, boolean>(); // Track terminal states

        // History for recurrent networks
        const tankStateHistory = new Map<number, Float32Array[]>();
        const MAX_HISTORY_LENGTH = 5; // Keep last 5 states for recurrent processing

        // initial game tick
        gameTick(TICK_TIME_SIMULATION);

        let steps = 0;
        let tankEids = query(world, [Tank, TankController, TankInputTensor]);

        // Initialize health tracking for all tanks
        for (let tankEid of tankEids) {
            mapTankToHealth.set(tankEid, TankInputTensor.health[tankEid]);
            mapTankToDone.set(tankEid, false);
            tankStateHistory.set(tankEid, []);
        }

        const stopInterval = macroTasks.addInterval(() => {
            steps++;

            // Terminate if episode is too long or all tanks are destroyed
            const onlyOneTankLeft = tankEids.length === 1;
            if (steps >= maxSteps || onlyOneTankLeft) {
                if (onlyOneTankLeft) {
                    // Reward for surviving
                    const survivorEid = tankEids[0];
                    const baseReward = mapTankToReward.get(survivorEid) || 0;
                    const survivalBonus = 5.0; // Существенный бонус за победу
                    mapTankToReward.set(survivorEid, baseReward + survivalBonus);

                    // Логируем победителя
                    console.log(`Tank ${ survivorEid } survived and got a bonus: ${ survivalBonus }`);
                }

                // Статистика эпизода
                const rewardStats = Object.entries(episodeRewardComponents).map(([key, values]) => {
                    const sum = values.reduce((a, b) => a + b, 0);
                    const avg = values.length > 0 ? sum / values.length : 0;
                    return `${ key }: ${ avg.toFixed(2) }`;
                }).join(', ');

                console.log(`Episode ended after ${ steps } steps with avg reward: ${
                    episodeReward.length > 0 ?
                        (episodeReward.reduce((a, b) => a + b, 0) / episodeReward.length).toFixed(2) :
                        'N/A'
                }`);
                console.log(`Reward components: ${ rewardStats }`);

                destroy();
                stopInterval();

                const totalReward = episodeReward.reduce((a, b) => a + b, 0);
                resolve(totalReward);
                return;
            }

            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;
            const vDelta = 200;
            const vWidth = width + vDelta;
            const vHeight = height + vDelta;
            const maxSpeed = 10_000;

            // PHASE 1: Process each tank for state evaluation and action determination
            for (let tankEid of tankEids) {
                // Initialize reward for this tank if not existing
                if (!mapTankToReward.has(tankEid)) {
                    mapTankToReward.set(tankEid, 0);
                }

                // Check if tank is in a terminal state (flag from previous tick)
                if (mapTankToDone.get(tankEid)) continue;

                // Prepare state tensor
                const vTankX = TankInputTensor.x[tankEid] + vDelta;
                const vTankY = TankInputTensor.y[tankEid] + vDelta;
                const inputVector = new Float32Array(INPUT_DIM);
                let k = 0;

                // Tank state
                inputVector[k++] = TankInputTensor.health[tankEid];
                inputVector[k++] = clamp(vTankX / vWidth, 0, 1);
                inputVector[k++] = clamp(vTankY / vHeight, 0, 1);
                inputVector[k++] = TankInputTensor.speed[tankEid] / maxSpeed;
                inputVector[k++] = TankInputTensor.rotation[tankEid] / Math.PI;
                inputVector[k++] = TankInputTensor.turretRotation[tankEid] / Math.PI;
                inputVector[k++] = TankInputTensor.projectileSpeed[tankEid] / maxSpeed;

                // Enemies data
                const enemiesBuffer = TankInputTensor.enemiesData.getBatche(tankEid);
                for (let i = 0; i < TANK_INPUT_TENSOR_MAX_ENEMIES; i++) {
                    enemiesBuffer[i * 4 + 0] = clamp(enemiesBuffer[i * 4 + 0] / vWidth, 0, 1);
                    enemiesBuffer[i * 4 + 1] = clamp(enemiesBuffer[i * 4 + 1] / vHeight, 0, 1);
                    enemiesBuffer[i * 4 + 2] = enemiesBuffer[i * 4 + 2] / maxSpeed;
                    enemiesBuffer[i * 4 + 3] = enemiesBuffer[i * 4 + 3] / maxSpeed;
                }
                inputVector.set(enemiesBuffer, k);
                k += enemiesBuffer.length;

                // Bullets data
                const bulletsBuffer = TankInputTensor.bulletsData.getBatche(tankEid);
                for (let i = 0; i < TANK_INPUT_TENSOR_MAX_BULLETS; i++) {
                    bulletsBuffer[i * 4 + 0] = clamp(bulletsBuffer[i * 4 + 0] / vWidth, 0, 1);
                    bulletsBuffer[i * 4 + 1] = clamp(bulletsBuffer[i * 4 + 1] / vHeight, 0, 1);
                    bulletsBuffer[i * 4 + 2] = bulletsBuffer[i * 4 + 2] / maxSpeed;
                    bulletsBuffer[i * 4 + 3] = bulletsBuffer[i * 4 + 3] / maxSpeed;
                }
                inputVector.set(bulletsBuffer, k);
                k += bulletsBuffer.length;

                // Normalize input using the agent's input normalizer
                const normalizedInputVector = agent.inputNormalizer.normalize(inputVector);

                // Update normalizer with new data
                agent.inputNormalizer.update(inputVector);

                // Keep track of history for this tank
                let history = tankStateHistory.get(tankEid);
                if (history) {
                    history.push(new Float32Array(normalizedInputVector));
                    // Limit history length
                    if (history.length > MAX_HISTORY_LENGTH) {
                        history.shift();
                    }
                    tankStateHistory.set(tankEid, history);
                }

                // Create tensor from normalized input vector
                const stateTensor = tf.tensor2d(
                    // @ts-ignore
                    [normalizedInputVector],
                );

                // Get action from policy
                const [actionTensor, logProbTensor] = agent.sampleAction(stateTensor);
                const valueTensor = agent.evaluateState(stateTensor);

                // Apply action to the game
                const actions = actionTensor.dataSync();

                const shouldShoot = actions[0] > 0;
                TankController.setShooting(tankEid, shouldShoot);
                TankController.setMove$(tankEid, actions[1]);
                TankController.setRotate$(tankEid, actions[2]);
                TankController.setTurretTarget$(
                    tankEid,
                    ((actions[3] + 1) / 2) * width,
                    ((actions[4] + 1) / 2) * height,
                );

                // Store tensors for later use after the game tick
                mapTankToState.set(tankEid, stateTensor);
                mapTankToAction.set(tankEid, actionTensor);
                mapTankToLogProb.set(tankEid, logProbTensor);
                mapTankToValue.set(tankEid, valueTensor);
            }

            // PHASE 2: Execute game tick after all controller updates
            gameTick(TICK_TIME_SIMULATION);

            // Get list of tanks after the game tick (some might be destroyed)
            const previousTankEids = tankEids;
            tankEids = [...query(world, [Tank, TankController, TankInputTensor])];

            // Find tanks that were destroyed in this tick
            const destroyedTanks = previousTankEids.filter(eid => !tankEids.includes(eid));

            // Add terminal states for destroyed tanks
            for (let destroyedEid of destroyedTanks) {
                mapTankToDone.set(destroyedEid, true);

                // Apply death penalty to destroyed tanks
                const prevReward = mapTankToReward.get(destroyedEid) || 0;
                const prevState = mapTankToState.get(destroyedEid);
                const prevAction = mapTankToAction.get(destroyedEid);
                const prevLogProb = mapTankToLogProb.get(destroyedEid);
                const prevValue = mapTankToValue.get(destroyedEid);

                const totalReward = prevReward - 3.0;
                mapTankToReward.set(destroyedEid, totalReward);

                // If we have a valid state-action pair from previous step, add it to the buffer
                if (prevState && prevAction && prevLogProb && prevValue) {
                    agent.storeExperience(
                        prevState,
                        prevAction,
                        prevLogProb,
                        totalReward,
                        prevValue,
                        true,
                    );

                    // Add to episode reward tracking
                    episodeReward.push(totalReward);
                }

                // If this is a terminal state, clean up the history
                tankStateHistory.delete(destroyedEid);

                console.log(`Tank ${ destroyedEid } was destroyed, terminal state applied`);
            }

            // PHASE 3: Calculate rewards and store experiences after the game tick
            for (let tankEid of tankEids) {
                // Get previous health
                const prevHealth = mapTankToHealth.get(tankEid) || 1.0;
                const currentHealth = TankInputTensor.health[tankEid];

                // Update health tracking
                mapTankToHealth.set(tankEid, currentHealth);

                // Get previous action for reward calculation
                const actions = mapTankToAction.get(tankEid)?.dataSync();

                // Calculate reward using the enhanced reward function
                const { totalReward, rewards } = calculateReward(
                    tankEid,
                    actions || [0, 0, 0, 0, 0],
                    prevHealth,
                    width,
                    height,
                );

                // Keep track of individual reward components for debugging
                Object.entries(rewards).forEach(([key, value]) => {
                    if (episodeRewardComponents[key]) {
                        episodeRewardComponents[key].push(value);
                    }
                });

                // Add to total reward for this tank
                const newTotalReward = (mapTankToReward.get(tankEid) || 0) + totalReward;
                mapTankToReward.set(tankEid, newTotalReward);

                // Get previous tensor states
                const prevState = mapTankToState.get(tankEid);
                const prevAction = mapTankToAction.get(tankEid);
                const prevLogProb = mapTankToLogProb.get(tankEid);
                const prevValue = mapTankToValue.get(tankEid);

                // If we have a valid state-action pair from previous step, add it to the buffer
                if (prevState && prevAction && prevLogProb && prevValue) {
                    agent.storeExperience(
                        prevState,
                        prevAction,
                        prevLogProb,
                        totalReward,
                        prevValue,
                        false,
                    );

                    // Add to episode reward tracking
                    episodeReward.push(totalReward);
                }
            }

            // // Report progress every 100 steps
            // if (steps % 100 === 0) {
            //     const activeCount = tankEids.length;
            //     const avgReward = episodeReward.length > 0 ?
            //         episodeReward.slice(-100).reduce((a, b) => a + b, 0) / Math.min(100, episodeReward.length) :
            //         0;
            //
            //     console.log(`Step ${ steps }: ${ activeCount } tanks active, recent avg reward: ${ avgReward.toFixed(2) }`);
            // }
        }, TICK_TIME_REAL);
    });
}

// Main enhanced PPO training function with better tracking and checkpointing
async function trainPPO(episodes: number = 100, checkpointInterval: number = 5): Promise<void> {
    console.log('Starting enhanced PPO training...');

    const agent = new PPOAgent();
    let episodesCompleted = 0;
    let totalReward = 0;
    let bestEpisodeReward = -Infinity;

    // Track metrics for visualization and analysis
    const trainingMetrics = {
        episodeRewards: [] as number[],
        actorLosses: [] as number[],
        criticLosses: [] as number[],
        avgRewards: [] as number[],
        episodeLengths: [] as number[],
    };

    // Create a tensorboard-like summary writer if available
    let summaryWriter: any = null;
    try {
        // This would be a custom class you implement or use a library
        // summaryWriter = new TensorboardSummaryWriter('tank-training-logs');
    } catch (error) {
        console.warn('Summary writer not available, continuing without logging');
    }

    try {
        // Try to load existing models
        const loaded = await agent.loadModels();
        if (loaded) {
            console.log('Loaded existing models, continuing training from episode', agent.episodeCount);
            episodesCompleted = agent.episodeCount;
        } else {
            console.log('Starting with new models');
        }

        // Training loop with improved checkpointing and error handling
        for (let i = episodesCompleted; i < episodesCompleted + episodes; i++) {
            console.log(`Starting episode ${ i + 1 }/${ episodesCompleted + episodes }`);

            // Track episode start time for performance monitoring
            const episodeStartTime = performance.now();

            try {
                // Run episode with a reasonable step limit
                const maxSteps = 5000; // Limit episode length
                const episodeReward = await runEpisode(agent, maxSteps);
                totalReward += episodeReward;

                // Track episode duration for metrics
                const episodeDuration = performance.now() - episodeStartTime;
                console.log(`Episode ${ i + 1 } completed in ${ (episodeDuration / 1000).toFixed(2) }s with reward: ${ episodeReward.toFixed(2) }`);

                // Update metrics
                trainingMetrics.episodeRewards.push(episodeReward);
                trainingMetrics.episodeLengths.push(maxSteps); // This will be actual steps if terminated early
                trainingMetrics.avgRewards.push(totalReward / (i + 1 - episodesCompleted));

                // Train after each episode
                const trainingStartTime = performance.now();

                // Track losses for this training session
                let actorLossSum = 0;
                let criticLossSum = 0;
                let trainingIterations = 0;

                // Multiple training iterations if enough data is available
                const minBatchesForTraining = 3; // Train only if we have at least 3x batch size data
                const minSamplesRequired = BATCH_SIZE * minBatchesForTraining;

                if (agent.buffer.size >= minSamplesRequired) {
                    // Number of training iterations based on buffer size
                    const trainIterations = Math.min(
                        Math.floor(agent.buffer.size / BATCH_SIZE),
                        5, // Cap maximum iterations per episode
                    );

                    for (let iter = 0; iter < trainIterations; iter++) {
                        // Train and get average losses
                        const { actorLoss, criticLoss } = await agent.train();

                        if (actorLoss !== undefined && criticLoss !== undefined) {
                            actorLossSum += actorLoss;
                            criticLossSum += criticLoss;
                            trainingIterations++;
                        }
                    }

                    // Log average losses
                    if (trainingIterations > 0) {
                        const avgActorLoss = actorLossSum / trainingIterations;
                        const avgCriticLoss = criticLossSum / trainingIterations;

                        trainingMetrics.actorLosses.push(avgActorLoss);
                        trainingMetrics.criticLosses.push(avgCriticLoss);

                        console.log(`Training complete (${ trainingIterations } iterations, ${ (performance.now() - trainingStartTime).toFixed(0) }ms)`);
                        console.log(`Average Actor Loss: ${ avgActorLoss.toFixed(4) }, Critic Loss: ${ avgCriticLoss.toFixed(4) }`);
                    } else {
                        console.log('No valid training occurred this episode');
                    }
                } else {
                    console.log(`Not enough samples for training: ${ agent.buffer.size }/${ minSamplesRequired }`);
                }

                // Update episode counter
                agent.episodeCount = i + 1;

                // Check if this is the best episode so far
                if (episodeReward > bestEpisodeReward) {
                    bestEpisodeReward = episodeReward;

                    // Save best model separately
                    await agent.saveModels('best');
                    console.log(`New best model saved with reward: ${ bestEpisodeReward.toFixed(2) }`);
                }

                // Regular checkpointing
                if ((i + 1) % checkpointInterval === 0) {
                    await agent.saveModels();
                    console.log(`Checkpoint saved at episode ${ i + 1 }`);

                    // Save metrics
                    try {
                        localStorage.setItem('tank-training-metrics', JSON.stringify(trainingMetrics));
                    } catch (error) {
                        console.warn('Failed to save metrics:', error);
                    }
                }

                // Log to tensorboard if available
                if (summaryWriter) {
                    summaryWriter.scalar('reward/episode', episodeReward, i);
                    summaryWriter.scalar('reward/average', totalReward / (i + 1 - episodesCompleted), i);
                    if (trainingIterations > 0) {
                        summaryWriter.scalar('loss/actor', actorLossSum / trainingIterations, i);
                        summaryWriter.scalar('loss/critic', criticLossSum / trainingIterations, i);
                    }
                    summaryWriter.scalar('training/buffer_size', agent.buffer.size, i);
                    summaryWriter.scalar('training/episode_length', maxSteps, i);
                    summaryWriter.flush();
                }

            } catch (error) {
                console.error(`Error in episode ${ i + 1 }:`, error);

                // Try to save current state before potential recovery
                try {
                    await agent.saveModels('recovery');
                    console.log('Recovery checkpoint saved');
                } catch (saveError) {
                    console.error('Failed to save recovery checkpoint:', saveError);
                }

                // Wait a moment before continuing to next episode
                await new Promise(resolve => setTimeout(resolve, 5000));
                 // Skip to next episode instead of crashing
            }
        }

        // Final save and cleanup
        await agent.saveModels();

        // Save final metrics
        try {
            localStorage.setItem('tank-training-metrics', JSON.stringify(trainingMetrics));

            // Optional: Generate and save summary plots/stats if needed
            // await generateTrainingSummary(trainingMetrics);
        } catch (error) {
            console.warn('Failed to save final metrics:', error);
        }

        console.log('Training completed successfully!');
        console.log(`Total episodes: ${ agent.episodeCount }`);
        console.log(`Best episode reward: ${ bestEpisodeReward.toFixed(2) }`);
        console.log(`Average reward: ${ (totalReward / episodes).toFixed(2) }`);

        if (summaryWriter) {
            summaryWriter.close();
        }

    } catch (error) {
        console.error('Critical error during training:', error);

        // Try emergency save
        try {
            await agent.saveModels('emergency');
            console.log('Emergency save completed');
        } catch (saveError) {
            console.error('Failed to perform emergency save:', saveError);
        }

        // Wait before potentially reloading page
        console.log('Reloading page in 10 seconds...');
        setTimeout(() => {
            window.location.reload();
        }, 10_000);
    }
}

trainPPO(1_000, 5);