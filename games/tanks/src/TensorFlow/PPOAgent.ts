import * as tf from '@tensorflow/tfjs';
import { LayersModel, Scalar } from '@tensorflow/tfjs';
import { createActorModel, createCriticModel, createExplorationBiasedActorModel } from './models.ts';
import { ACTION_DIM, INPUT_DIM } from './consts.ts';

const PPO_EPOCHS = 8;
export const BATCH_SIZE = 128;
const CLIP_EPSILON = 0.3;
const ENTROPY_WEIGHTS = [0.05, 0.02, 0.02, 0.01, 0.01];
const GAMMA = 0.99; // Discount factor
const LAMBDA = 0.95; // GAE parameter

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

    constructor(private capacity: number = 1024 * 4) {
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
    computeReturnsAndAdvantages(gamma = GAMMA, lambda = LAMBDA): [tf.Tensor, tf.Tensor] {
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
            const delta = reward + (done ? 0 : gamma * nextStateValue) - value;
            nextAdvantage = delta + (done ? 0 : gamma * lambda * nextAdvantage);
            nextReturn = reward + (done ? 0 : gamma * nextReturn);

            returns[i] = nextReturn;
            advantages[i] = nextAdvantage;
        }

        // Normalize advantages
        let advantageSum = 0;
        let advantageSqSum = 0;
        for (let i = 0; i < advantages.length; i++) {
            advantageSum += advantages[i];
        }
        const advantageMean = advantageSum / advantages.length;
        for (let i = 0; i < advantages.length; i++) {
            advantageSqSum += Math.pow(advantages[i] - advantageMean, 2);
        }
        const advantageStd = Math.sqrt(advantageSqSum / advantages.length) + 1e-8;
        const normalizedAdvantages = advantages.map(adv => (adv - advantageMean) / advantageStd);

        // Normalize returns
        let returnSum = 0;
        let returnSqSum = 0;
        for (let i = 0; i < returns.length; i++) {
            returnSum += returns[i];
        }
        const returnMean = returnSum / returns.length;
        for (let i = 0; i < returns.length; i++) {
            returnSqSum += Math.pow(returns[i] - returnMean, 2);
        }
        const returnStd = Math.sqrt(returnSqSum / returns.length) + 1e-8;
        const normalizedReturns = returns.map(ret => (ret - returnMean) / returnStd);

        return [
            tf.tensor1d(normalizedReturns),
            tf.tensor1d(normalizedAdvantages),
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

// Implementation of improved PPO Agent
export class PPOAgent {
    actorMean: LayersModel;
    actorStd: LayersModel;
    critic: LayersModel;
    buffer: PrioritizedExperienceBuffer;
    inputNormalizer: InputNormalizer;

    constructor(useExplorationBias: boolean = false) {
        if (useExplorationBias) {
            const { meanModel, stdModel } = createExplorationBiasedActorModel();
            this.actorMean = meanModel;
            this.actorStd = stdModel;
        } else {
            const { meanModel, stdModel } = createActorModel();
            this.actorMean = meanModel;
            this.actorStd = stdModel;
        }

        this.critic = createCriticModel();
        this.buffer = new PrioritizedExperienceBuffer();
        this.inputNormalizer = new InputNormalizer(INPUT_DIM);
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
    async train(learningRate: number) {
        if (this.buffer.size < BATCH_SIZE) {
            console.log('Not enough samples for training');
            return {};
        }

        // Динамически определяем скорость обучения на основе номера эпизода
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
            console.log(`Models and input statistics loaded successfully (version: ${ version })`);
            return true;
        } catch (error) {
            console.log(`Failed to load models (version: ${ version }):`, error);
            return false;
        }
    }
}