import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { INPUT_DIM } from '../Common/consts';
import { getCurrentExperiment, RLExperimentConfig } from './experiment-config';

// Experience replay buffer for reinforcement learning
class ReplayMemory {
    private buffer: Array<{
        state: tf.Tensor;
        action: tf.Tensor;
        reward: number;
        nextState: tf.Tensor;
        done: boolean;
    }>;
    private capacity: number;

    constructor(capacity: number) {
        this.buffer = [];
        this.capacity = capacity;
    }

    add(state: tf.Tensor, action: tf.Tensor, reward: number, nextState: tf.Tensor, done: boolean) {
        if (this.buffer.length >= this.capacity) {
            // Remove oldest experience and dispose tensors to prevent memory leaks
            const oldest = this.buffer.shift();
            oldest?.state.dispose();
            oldest?.action.dispose();
            oldest?.nextState.dispose();
        }

        // Add new experience
        this.buffer.push({
            state: state.clone(),
            action: action.clone(),
            reward,
            nextState: nextState.clone(),
            done,
        });
    }

    sample(batchSize: number) {
        if (this.buffer.length < batchSize) {
            return null;
        }

        const samples = [];
        const indices = new Set<number>();

        // Randomly select experiences
        while (indices.size < batchSize) {
            const index = Math.floor(Math.random() * this.buffer.length);
            if (!indices.has(index)) {
                indices.add(index);
                samples.push(this.buffer[index]);
            }
        }

        // Batch tensors for efficient processing
        const states = tf.stack(samples.map(sample => sample.state));
        const actions = tf.stack(samples.map(sample => sample.action));
        const rewards = tf.tensor1d(samples.map(sample => sample.reward));
        const nextStates = tf.stack(samples.map(sample => sample.nextState));
        const dones = tf.tensor1d(samples.map(sample => sample.done ? 1 : 0));

        return {
            states,
            actions,
            rewards,
            nextStates,
            dones,
        };
    }

    size() {
        return this.buffer.length;
    }

    // Clean up tensors when not needed anymore
    dispose() {
        this.buffer.forEach(experience => {
            experience.state.dispose();
            experience.action.dispose();
            experience.nextState.dispose();
        });
        this.buffer = [];
    }
}

// Shared RL Agent for all tanks
export class SharedTankRLAgent {
    private memory: ReplayMemory;
    private model: tf.LayersModel;
    private targetModel: tf.LayersModel;
    private epsilon: number;
    private episodeCount: number;
    private optimizer: tf.Optimizer;
    private logger: {
        episodeRewards: number[];
        episodeLengths: number[];
        epsilon: number[];
        losses: number[];
    };
    private updateCounter: number = 0;
    private config: RLExperimentConfig;

    constructor() {
        // Get current experiment config
        this.config = getCurrentExperiment();

        // Initialize with config values
        this.memory = new ReplayMemory(this.config.memorySize);
        this.epsilon = this.config.epsilon;
        this.episodeCount = 0;
        this.optimizer = tf.train.adam(this.config.learningRate);

        // Initialize logger
        this.logger = {
            episodeRewards: [],
            episodeLengths: [],
            epsilon: [],
            losses: [],
        };

        // Create models
        this.model = this.createModel();
        this.targetModel = this.createModel();

        // Initialize target model with the same weights
        this.updateTargetModel();

        console.log(`Shared RL Agent initialized with experiment: ${ this.config.name }`);
    }

    // Remember experience in replay buffer
    remember(state: tf.Tensor, action: tf.Tensor, reward: number, nextState: tf.Tensor, done: boolean) {
        this.memory.add(state, action, reward, nextState, done);
    }

    // Choose action using epsilon-greedy policy
    act(state: tf.Tensor, isTraining: boolean = true): tf.Tensor {
        if (isTraining && Math.random() < this.epsilon) {
            // Exploration: random action
            return tf.tidy(() => {
                const shootRandom = tf.tensor1d([Math.random() > 0.5 ? 1 : 0]);
                const moveRandom = tf.randomUniform([2], -1, 1);
                const aimRandom = tf.randomUniform([2], -1, 1);
                return tf.concat([shootRandom, moveRandom, aimRandom], 0);
            });
        } else {
            // Exploitation: use model prediction
            return tf.tidy(() => {
                const stateTensor = state.expandDims(0);
                const predictions = this.model.predict(stateTensor) as tf.Tensor[];

                // Extract output components
                // Обратите внимание, что мы сначала делаем squeeze(), а затем берем первый элемент,
                // чтобы получить скаляр для shootProb и 1D тензоры для координат

                // Для стрельбы - получаем одно значение (сначала squeeze, затем arraySync)
                const shootPrediction = predictions[0].squeeze(); // [1]
                const shootProb = shootPrediction.arraySync() as number;
                const shootAction = tf.tensor1d([shootProb > 0.5 ? 1 : 0]);

                // Для движения - получаем два значения как 1D тензор
                const moveCoords = predictions[1].squeeze(); // [2]

                // Для прицеливания - получаем два значения как 1D тензор
                const aimCoords = predictions[2].squeeze(); // [2]
                // Объединяем все компоненты
                return tf.concat([shootAction, moveCoords, aimCoords], 0);
            });
        }
    }

    // Train the model on a batch from replay memory
    async train() {
        if (this.memory.size() < this.config.batchSize) {
            return 0; // Not enough samples
        }

        const batch = this.memory.sample(this.config.batchSize);
        if (!batch) return 0;

        const { states, actions, rewards, nextStates, dones } = batch;

        let totalLoss = 0;

        try {
            // Вычисляем целевые значения для стрельбы (Q-learning)
            const shootTargets = tf.tidy(() => {
                // Получаем предсказания для следующих состояний
                const targetPredictions = this.targetModel.predict(nextStates) as tf.Tensor[];
                const nextShootValues = targetPredictions[0]; // Выходы для стрельбы

                // Маска для терминальных состояний: 1 - done
                // dones содержит 1 для терминальных состояний, 0 для нетерминальных
                // Поэтому инвертируем: 1 - done
                const notDoneMask = tf.scalar(1).sub(dones);

                // Q-target = reward + γ * Q(s', a') * (1 - done)
                // 1. Умножаем Q(s', a') на gamma
                const discountedNextValues = nextShootValues.mul(tf.scalar(this.config.gamma));

                // 2. Учитываем терминальные состояния
                const futureValues = discountedNextValues.mul(notDoneMask.expandDims(1));

                // 3. Считаем конечные целевые значения
                return rewards.expandDims(1).add(futureValues);
            });

            // Извлекаем компоненты действий
            const moveActions = actions.slice([0, 1], [this.config.batchSize, 2]);
            const aimActions = actions.slice([0, 3], [this.config.batchSize, 2]);

            // Вызываем trainOnBatch с правильно сформированными целевыми тензорами
            const result = await this.model.trainOnBatch(
                states,
                [shootTargets, moveActions, aimActions],
            );

            // Вычисляем общую потерю
            if (Array.isArray(result)) {
                totalLoss = result.reduce((sum, loss) => sum + loss, 0);
            } else {
                totalLoss = result;
            }

            // Логируем потерю
            this.logger.losses.push(totalLoss);

            // Обновляем целевую сеть при необходимости
            this.updateCounter++;
            if (this.updateCounter >= this.config.updateTargetEvery) {
                this.updateTargetModel();
                this.updateCounter = 0;
            }

            // Освобождаем созданные тензоры
            shootTargets.dispose();

            return totalLoss;
        } catch (error) {
            console.error('Error during training:', error);
            return 0;
        } finally {
            // Освобождаем тензоры батча
            states.dispose();
            actions.dispose();
            rewards.dispose();
            nextStates.dispose();
            dones.dispose();
        }
    }

    // Decay epsilon value for exploration-exploitation balance
    updateEpsilon() {
        if (this.epsilon > this.config.epsilonMin) {
            this.epsilon *= this.config.epsilonDecay;
        }
        this.logger.epsilon.push(this.epsilon);
    }

    // Log episode data for monitoring
    logEpisode(reward: number, length: number) {
        this.logger.episodeRewards.push(reward);
        this.logger.episodeLengths.push(length);
        this.episodeCount++;

        // Log stats every 10 episodes
        if (this.episodeCount % 10 === 0) {
            const last10Rewards = this.logger.episodeRewards.slice(-10);
            const avgReward = last10Rewards.reduce((a, b) => a + b, 0) / 10;

            console.log(`Episode: ${ this.episodeCount }`);
            console.log(`Average Reward (last 10): ${ avgReward.toFixed(2) }`);
            console.log(`Epsilon: ${ this.epsilon.toFixed(4) }`);
            console.log(`Memory size: ${ this.memory.size() }`);

            if (this.logger.losses.length > 0) {
                const last10Losses = this.logger.losses.slice(-10);
                const avgLoss = last10Losses.reduce((a, b) => a + b, 0) / last10Losses.length;
                console.log(`Average Loss: ${ avgLoss.toFixed(4) }`);
            }
        }

        // Save model periodically
        if (this.episodeCount % this.config.saveModelEvery === 0) {
            this.saveModel();
        }
    }

    // Get current stats for display/logging
    getStats() {
        const last10Rewards = this.logger.episodeRewards.slice(-10);
        const avgReward = last10Rewards.length > 0
            ? last10Rewards.reduce((a, b) => a + b, 0) / last10Rewards.length
            : 0;

        const last10Losses = this.logger.losses.slice(-10);
        const avgLoss = last10Losses.length > 0
            ? last10Losses.reduce((a, b) => a + b, 0) / last10Losses.length
            : 0;

        return {
            episodeCount: this.episodeCount,
            epsilon: this.epsilon,
            memorySize: this.memory.size(),
            avgReward: avgReward,
            avgLoss: avgLoss,
            experimentName: this.config.name,
        };
    }

    // Save model to storage
    async saveModel() {
        try {
            await this.model.save('indexeddb://tank-rl-shared-model');
            localStorage.setItem('tank-rl-state', JSON.stringify({
                epsilon: this.epsilon,
                episodeCount: this.episodeCount,
                updateCounter: this.updateCounter,
                experimentName: this.config.name,
                timestamp: new Date().toISOString(),
                losses: this.logger.losses.slice(-10), // Последние 10 значений потерь
                rewards: this.logger.episodeRewards.slice(-10), // Последние 10 наград
            }));
            console.log(`Shared model saved at episode ${ this.episodeCount }`);
            return true;
        } catch (error) {
            console.error('Error saving model:', error);
            return false;
        }
    }

    // Load model from storage
    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('indexeddb://tank-rl-shared-model');
            this.model.compile({
                optimizer: this.optimizer,
                loss: {
                    shoot_output: 'binaryCrossentropy',
                    move_output: 'meanSquaredError',
                    aim_output: 'meanSquaredError',
                },
            });
            this.updateTargetModel();
            console.log('Shared model loaded successfully');

            const metadata = JSON.parse(localStorage.getItem('tank-rl-state') ?? '{}');
            if (metadata) {
                this.epsilon = metadata.epsilon || this.config.epsilon;
                this.episodeCount = metadata.episodeCount || 0;
                this.updateCounter = metadata.updateCounter || 0;

                // Загружаем историю потерь и наград, если она есть
                if (metadata.losses && metadata.losses.length > 0) {
                    this.logger.losses = metadata.losses;
                }

                if (metadata.rewards && metadata.rewards.length > 0) {
                    this.logger.episodeRewards = metadata.rewards;
                }

                console.log(`Restored training state: episode ${ this.episodeCount }, epsilon ${ this.epsilon.toFixed(4) }`);

                // Проверяем, не изменился ли эксперимент
                if (metadata.experimentName && metadata.experimentName !== this.config.name) {
                    console.warn(`Loaded model was trained with experiment "${ metadata.experimentName }", but current experiment is "${ this.config.name }"`);
                }
            }
            return true;
        } catch (error) {
            console.warn('Could not load shared model, starting with a new one:', error);
            return false;
        }
    }

    // Dispose tensors to prevent memory leaks
    dispose() {
        this.model.dispose();
        this.targetModel.dispose();
        this.memory.dispose();
        console.log('Shared RL Agent resources disposed');
    }

    private createModel(): tf.LayersModel {
        // Input layer
        const input = tf.layers.input({ shape: [INPUT_DIM] });

        // Shared layers based on experiment config
        let shared = input;
        for (const units of this.config.hiddenLayers) {
            shared = tf.layers.dense({
                units,
                activation: 'relu',
                kernelInitializer: 'glorotUniform',
            }).apply(shared) as tf.SymbolicTensor;
        }

        // Output heads

        // 1. Shoot head - binary decision with probability
        const shootHead = tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            name: 'shoot_output',
        }).apply(shared) as tf.SymbolicTensor;

        // 2. Movement head - target coordinates (x, y)
        const moveHead = tf.layers.dense({
            units: 2,
            activation: 'tanh',  // Output in range [-1, 1]
            name: 'move_output',
        }).apply(shared) as tf.SymbolicTensor;

        // 3. Aim head - target coordinates (x, y)
        const aimHead = tf.layers.dense({
            units: 2,
            activation: 'tanh',  // Output in range [-1, 1]
            name: 'aim_output',
        }).apply(shared) as tf.SymbolicTensor;

        // Create model with multiple outputs
        const model = tf.model({
            inputs: input,
            outputs: [shootHead, moveHead, aimHead],
        });

        // Compile model
        model.compile({
            optimizer: this.optimizer,
            loss: {
                shoot_output: 'binaryCrossentropy',
                move_output: 'meanSquaredError',
                aim_output: 'meanSquaredError',
            },
        });

        return model;
    }

    // Update target network with current model weights
    private updateTargetModel() {
        this.targetModel.setWeights(this.model.getWeights());
    }
}

// Create a singleton instance of the shared agent
let sharedAgent: SharedTankRLAgent | null = null;

// Get or create the shared agent
export function getSharedAgent(): SharedTankRLAgent {
    if (!sharedAgent) {
        sharedAgent = new SharedTankRLAgent();
    }
    return sharedAgent;
}

// Clean up the shared agent
export function disposeSharedAgent() {
    if (sharedAgent) {
        sharedAgent.dispose();
        sharedAgent = null;
    }
}