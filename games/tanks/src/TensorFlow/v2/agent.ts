import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { INPUT_DIM, TANK_RADIUS } from '../Common/consts';
import { getCurrentExperiment, RLExperimentConfig } from './experiment-config';
import { ComponentRewards } from '../Common/calculateMultiHeadReward.ts';
import { smoothstep } from '../../../../../lib/math.ts';
import {
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
    TankInputTensor,
} from '../../ECS/Components/TankState.ts';
import { random, randomSign } from '../../../../../lib/random.ts';

// Experience replay buffer for reinforcement learning
class ReplayMemory {
    private buffer: Array<{
        state: tf.Tensor;
        action: tf.Tensor;
        reward: ComponentRewards;
        nextState: tf.Tensor;
        done: boolean;
    }>;
    private capacity: number;

    constructor(capacity: number) {
        this.buffer = [];
        this.capacity = capacity;
    }

    add(state: tf.Tensor, action: tf.Tensor, reward: ComponentRewards, nextState: tf.Tensor, done: boolean) {
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
            const index = Math.floor(random() * this.buffer.length);
            if (!indices.has(index)) {
                indices.add(index);
                samples.push(this.buffer[index]);
            }
        }

        // Batch tensors for efficient processing
        const states = tf.stack(samples.map(sample => sample.state));
        const nextStates = tf.stack(samples.map(sample => sample.nextState));
        const actions = tf.stack(samples.map(sample => sample.action));
        const dones = tf.tensor1d(samples.map(sample => sample.done ? 1 : 0));
        const rewards = {
            shoot: tf.tensor1d(samples.map(sample => sample.reward.shoot.total)),
            movement: tf.tensor1d(samples.map(sample => sample.reward.movement.total)),
            aim: tf.tensor1d(samples.map(sample => sample.reward.aim.total)),
            common: tf.tensor1d(samples.map(sample => sample.reward.common.total)),
            total: tf.tensor1d(samples.map(sample => sample.reward.totalReward)),
        };

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
        losses: {
            total: number[];
            shoot: number[];
            movement: number[];
            aim: number[];
        };
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
            losses: {
                total: [],
                shoot: [],
                movement: [],
                aim: [],
            },
        };

        // Create models
        this.model = this.createModel();
        this.targetModel = this.createModel();

        // Initialize target model with the same weights
        this.updateTargetModel();

        console.log(`Shared RL Agent initialized with experiment: ${ this.config.name }`);
    }

    // Remember experience in replay buffer
    remember(state: tf.Tensor, action: tf.Tensor, reward: ComponentRewards, nextState: tf.Tensor, done: boolean) {
        this.memory.add(state, action, reward, nextState, done);
    }

    /**
     * Выбор действия с использованием смешанной стратегии исследования/эксплуатации
     * @param state Состояние танка
     * @param tankId ID танка
     * @param isTraining Находимся ли мы в режиме обучения
     * @returns Тензор с действием [shoot, move, rotate, aimX, aimY]
     */
    act(state: tf.Tensor, tankId: number, isTraining: boolean = true): number[] {
        // Если мы не в режиме обучения, просто используем модель
        if (!isTraining) {
            return this.useModelPrediction(state);
        }

        // Выбираем стратегию исследования/эксплуатации на основе epsilon
        const strategy = selectStrategy(this.epsilon, this.config.epsilon, this.config.epsilonMin);

        switch (strategy) {
            case 'pure_random':
                // Полностью случайные действия (настоящий exploration)
                return generatePureRandomAction();
            case 'guided_random':
                // Управляемое случайное исследование (guided exploration)
                return generateGuidedRandomAction(tankId);
            case 'exploitation':
            default:
                // Используем обученную модель (exploitation)
                return this.useModelPrediction(state);
        }
    }

    async train() {
        if (this.memory.size() < this.config.batchSize) {
            return { total: 0, shoot: 0, movement: 0, aim: 0 }; // Недостаточно примеров
        }

        const batch = this.memory.sample(this.config.batchSize);
        if (!batch) return { total: 0, shoot: 0, movement: 0, aim: 0 };

        const { states, actions, rewards, nextStates, dones } = batch;

        let losses = {
            total: 0,
            shoot: 0,
            movement: 0,
            aim: 0,
        };

        // Временные тензоры, которые нужно будет освободить
        const tempTensors: tf.Tensor[] = [];

        try {
            // Получаем предсказания модели для текущих состояний
            const currentPredictions = this.model.predict(states) as tf.Tensor[];
            tempTensors.push(...currentPredictions);

            // Получаем предсказания target-модели для следующих состояний
            const targetPredictions = this.targetModel.predict(nextStates) as tf.Tensor[];
            tempTensors.push(...targetPredictions);

            // Маска для «не терминальных» состояний: (1 - done)
            const notDoneMask = tf.scalar(1).sub(dones).expandDims(1);
            tempTensors.push(notDoneMask);

            // ========= Формируем Q-таргеты для всех трёх голов ==========

            // 1) Стрельба
            // Q-target: reward.shoot + γ * Q'(след.состояние, shoot) * (1 - done)
            const futureShootRewards = targetPredictions[0]
                .mul(tf.scalar(this.config.gamma))
                .mul(notDoneMask);
            tempTensors.push(futureShootRewards);

            const shootTargets = rewards.shoot.expandDims(1).add(futureShootRewards);
            tempTensors.push(shootTargets);

            // 2) Движение
            // Q-target: reward.movement + γ * Q'(след.состояние, move) * (1 - done)
            const futureMoveRewards = targetPredictions[1]
                .mul(tf.scalar(this.config.gamma))
                .mul(notDoneMask);
            tempTensors.push(futureMoveRewards);

            const moveTargets = rewards.movement.expandDims(1).add(futureMoveRewards);
            tempTensors.push(moveTargets);

            // 3) Прицеливание
            // Q-target: reward.aim + γ * Q'(след.состояние, aim) * (1 - done)
            const futureAimRewards = targetPredictions[2]
                .mul(tf.scalar(this.config.gamma))
                .mul(notDoneMask);
            tempTensors.push(futureAimRewards);

            const aimTargets = rewards.aim.expandDims(1).add(futureAimRewards);
            tempTensors.push(aimTargets);

            // ========= Запускаем ЕДИНЫЙ trainOnBatch =========
            // Каждая голова получает свой таргет, обучаемся за один проход.
            // При этом лосс-сумма (или среднее) пойдёт в обратное распространение сразу по всем выходам.

            const trainResult = await this.model.trainOnBatch(
                states,
                {
                    shoot_output: shootTargets,
                    move_output: moveTargets,
                    aim_output: aimTargets,
                },
            );

            // Обычно tfjs при мульти-выходе возвращает массив вида:
            // [totalLoss, shootLoss, moveLoss, aimLoss] (или схожим порядком).
            // Чтобы понять точный порядок, смотрите документацию или логи от tfjs при компиляции.
            // Здесь предполагаем, что "нулевой" элемент – totalLoss, дальше идут потери по головам.
            if (Array.isArray(trainResult)) {
                // Пример: trainResult = [totalLoss, shootLoss, moveLoss, aimLoss]
                // Будьте внимательны, в некоторых версиях порядок бывает другой.
                const [totalLoss, shootLoss, moveLoss, aimLoss] = trainResult;
                losses = {
                    total: totalLoss,
                    shoot: shootLoss,
                    movement: moveLoss,
                    aim: aimLoss,
                };
            } else {
                // Если почему-то вернулся не массив (например, если у вас один выход),
                // то подставляем всё в total
                losses.total = trainResult as number;
            }

            // Логирование
            if (!this.logger.losses || !this.logger.losses.total) {
                this.logger.losses = {
                    total: [],
                    shoot: [],
                    movement: [],
                    aim: [],
                };
            }
            this.logger.losses.total.push(losses.total);
            this.logger.losses.shoot.push(losses.shoot);
            this.logger.losses.movement.push(losses.movement);
            this.logger.losses.aim.push(losses.aim);

            // Обновление target-модели, если достигли нужного счётчика
            this.updateCounter++;
            if (this.updateCounter >= this.config.updateTargetEvery) {
                this.updateTargetModel();
                this.updateCounter = 0;
            }

            return losses;
        } catch (error) {
            console.error('Error during multi-head training:', error);
            return { total: 0, shoot: 0, movement: 0, aim: 0 };
        } finally {
            // Освобождаем временные тензоры
            tempTensors.forEach(tensor => {
                if (tensor && tensor.dispose) {
                    tensor.dispose();
                }
            });

            // Освобождаем тензоры батча
            states.dispose();
            actions.dispose();
            rewards.shoot.dispose();
            rewards.movement.dispose();
            rewards.aim.dispose();
            rewards.common.dispose();
            rewards.total.dispose();
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
    // Log episode data for monitoring
    logEpisode(reward: number, length: number) {
        this.logger.episodeRewards.push(reward);
        this.logger.episodeLengths.push(length);
        this.episodeCount++;

        // Log stats every 10 episodes
        if (this.episodeCount % 10 === 0) {
            // Вычисляем средние награды
            const last10Rewards = this.logger.episodeRewards.slice(-10);
            const avgReward = last10Rewards.reduce((a, b) => a + b, 0) /
                Math.max(1, last10Rewards.length);

            // Проверяем, что у нас есть логи потерь
            if (!this.logger.losses.total) {
                this.logger.losses = {
                    total: [],
                    shoot: [],
                    movement: [],
                    aim: [],
                };
            }

            // Вычисляем средние потери по компонентам
            const avgTotalLoss = this.logger.losses.total.length > 0
                ? this.logger.losses.total.slice(-10).reduce((a, b) => a + b, 0) /
                Math.max(1, Math.min(10, this.logger.losses.total.length))
                : 0;

            const avgShootLoss = this.logger.losses.shoot.length > 0
                ? this.logger.losses.shoot.slice(-10).reduce((a, b) => a + b, 0) /
                Math.max(1, Math.min(10, this.logger.losses.shoot.length))
                : 0;

            const avgMoveLoss = this.logger.losses.movement.length > 0
                ? this.logger.losses.movement.slice(-10).reduce((a, b) => a + b, 0) /
                Math.max(1, Math.min(10, this.logger.losses.movement.length))
                : 0;

            const avgAimLoss = this.logger.losses.aim.length > 0
                ? this.logger.losses.aim.slice(-10).reduce((a, b) => a + b, 0) /
                Math.max(1, Math.min(10, this.logger.losses.aim.length))
                : 0;

            // Выводим информацию о тренировке
            console.log(`Episode: ${ this.episodeCount }`);
            console.log(`Average Reward (last 10): ${ avgReward.toFixed(2) }`);
            console.log(`Epsilon: ${ this.epsilon.toFixed(4) }`);
            console.log(`Memory size: ${ this.memory.size() }`);
            console.log('Average Losses:');
            console.log(`  Total: ${ avgTotalLoss.toFixed(3) }`);
            console.log(`  Shoot: ${ avgShootLoss.toFixed(3) }`);
            console.log(`  Movement: ${ avgMoveLoss.toFixed(3) }`);
            console.log(`  Aim: ${ avgAimLoss.toFixed(3) }`);
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

        const last10Losses = this.logger.losses.total.slice(-10);
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
                losses: {
                    aim: this.logger.losses.aim.slice(-10),
                    shoot: this.logger.losses.shoot.slice(-10),
                    movement: this.logger.losses.movement.slice(-10),
                    total: this.logger.losses.total.slice(-10),
                }, // Потери за последние 10 эпизодов
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
            this.compile(this.model);
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

        this.compile(model);

        return model;
    }

    private compile(model: tf.LayersModel) {
        // Compile model
        model.compile({
            optimizer: this.optimizer,
            loss: {
                shoot_output: 'binaryCrossentropy',
                move_output: 'meanSquaredError',
                aim_output: 'meanSquaredError',
            },
        });
    }

    // Update target network with current model weights
    private updateTargetModel() {
        this.targetModel.setWeights(this.model.getWeights());
    }

    private useModelPrediction(state: tf.Tensor): number[] {
        return tf.tidy(() => {
            const stateTensor = state.expandDims(0);
            const predictions = this.model.predict(stateTensor) as tf.Tensor[];

            // Извлекаем выходы модели
            const shootProbability = predictions[0].squeeze().arraySync() as number;
            const moveValues = predictions[1].squeeze().arraySync() as number[];
            const aimValues = predictions[2].squeeze().arraySync() as number[];
            // Преобразуем вероятность стрельбы в бинарное действие
            const shootAction = shootProbability > 0.5 ? 1 : 0;

            // Создаем итоговый тензор с действиями
            return [shootAction, ...moveValues, ...aimValues];
        });
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

/**
 * Выбор стратегии на основе текущего значения epsilon и других факторов
 * @returns Выбранная стратегия
 */
function selectStrategy(epsilon: number, epsilonBase: number, epsilonMin: number): 'pure_random' | 'guided_random' | 'exploitation' {
    // По мере уменьшения epsilon, мы снижаем вероятность случайных действий
    // и увеличиваем вероятность использования модели

    // Начальное распределение:
    // - 40% полностью случайные
    // - 40% управляемые случайные
    // - 20% модель

    // Конечное распределение:
    // - 5% полностью случайные
    // - 10% управляемые случайные
    // - 85% модель

    // Вычисляем, как далеко мы продвинулись в обучении (от 0 до 1)
    const progress = 1 - ((epsilon - epsilonMin) /
        (epsilonBase - epsilonMin));

    // Вычисляем вероятности каждой стратегии
    const pureRandomProb = 0.4 - 0.35 * progress; // от 40% до 5%
    const guidedRandomProb = 0.4 - 0.3 * progress; // от 40% до 10%
    // exploitationProb = 1 - pureRandomProb - guidedRandomProb; // от 20% до 85%

    // Выбираем стратегию на основе вероятностей
    const rand = random();
    if (rand < pureRandomProb) {
        return 'pure_random';
    } else if (rand < pureRandomProb + guidedRandomProb) {
        return 'guided_random';
    } else {
        return 'exploitation';
    }
}

/**
 * Генерация управляемых случайных действий с элементами простых стратегий
 * @param tankId ID танка
 * @returns Массив с действием [shoot, move, rotate, aimX, aimY]
 */
function generateGuidedRandomAction(tankId: number): number[] {
    // Получаем данные о танке
    const tankX = TankInputTensor.x[tankId];
    const tankY = TankInputTensor.y[tankId];

    // Базовые случайные значения для движения и вращения
    let moveValue = (random() * 1.6 - 0.8); // от -0.8 до 0.8
    let rotateValue = (random() * 1.6 - 0.8); // от -0.8 до 0.8

    // Значения для прицеливания
    let aimX = 0;
    let aimY = 0;

    // Случайно выбираем стратегию поведения
    const behaviorStrategy = random();

    if (behaviorStrategy < 0.4) {
        // Стратегия: поиск и отслеживание ближайшего врага
        const nearestEnemy = findNearestEnemy(tankId);

        if (nearestEnemy) {
            // Устанавливаем прицел на врага с небольшим случайным отклонением
            aimX = nearestEnemy.x + (random() * 0.3 - 0.15);
            aimY = nearestEnemy.y + (random() * 0.3 - 0.15);

            // Вычисляем направление к врагу для движения
            const distToEnemy = Math.hypot(nearestEnemy.x - tankX, nearestEnemy.y - tankY);

            // Если враг слишком близко, отъезжаем, иначе подъезжаем на среднюю дистанцию
            if (distToEnemy < TANK_RADIUS * 3) {
                // Отъезжаем от врага
                moveValue = -0.5 - random() * 0.5; // от -0.5 до -1.0
            } else if (distToEnemy > TANK_RADIUS * 10) {
                // Подъезжаем к врагу
                moveValue = 0.5 + random() * 0.5; // от 0.5 до 1.0
            } else {
                // Поддерживаем дистанцию, случайное движение
                moveValue = random() * 1.6 - 0.8; // от -0.8 до 0.8
            }

            // Стреляем с большей вероятностью при хорошем прицеливании
            const shootValue = random() < 0.7 ? 1 : 0;

            return [shootValue, moveValue, rotateValue, aimX, aimY];
        }
    } else if (behaviorStrategy < 0.7) {
        // Стратегия: патрулирование с периодическим сканированием
        // Движемся и поворачиваемся более последовательно
        moveValue = 0.5 + random() * 0.5; // от 0.5 до 1.0
        rotateValue = (random() - 0.5) * 0.6; // от -0.3 до 0.3 (меньше поворотов)

        // Прицеливаемся в разные части карты, но более плавно
        aimX = random() * 2 - 1;
        aimY = random() * 2 - 1;

        // Меньше стреляем при патрулировании
        const shootValue = random() < 0.3 ? 1 : 0;

        return [shootValue, moveValue, rotateValue, aimX, aimY];
    } else {
        // Стратегия: случайное движение, но сканирование потенциальных угроз
        // Ищем ближайшую пулю и уворачиваемся
        const nearestBullet = findNearestBullet(tankId);

        if (nearestBullet && nearestBullet.danger > 0.3) {
            // Есть опасная пуля - уворачиваемся

            // Выбираем направление уворота случайно
            const dodgeDir = random() > 0.5 ? 1 : -1;

            // Ставим более высокую скорость для уворота
            moveValue = 0.8 + random() * 0.2; // от 0.8 до 1.0

            // Поворачиваем в направлении движения
            rotateValue = dodgeDir * (0.5 + random() * 0.5); // от 0.5 до 1.0 с нужным знаком

            // Прицеливаемся в направлении, откуда пришла пуля
            aimX = nearestBullet.x + nearestBullet.vx * -5; // -5 это коэффициент "заглядывания назад"
            aimY = nearestBullet.y + nearestBullet.vy * -5;

            // Нормализуем координаты прицела
            const aimLen = Math.hypot(aimX, aimY);
            if (aimLen > 0) {
                aimX = aimX / aimLen;
                aimY = aimY / aimLen;
            }

            // Редко стреляем при уворачивании
            const shootValue = random() < 0.2 ? 1 : 0;

            return [shootValue, moveValue, rotateValue, aimX, aimY];
        } else {
            // Нет опасности - случайные действия с небольшим смещением в сторону разумности
            moveValue = random() * 1.6 - 0.3; // от -0.3 до 1.3 (смещение вперёд)
            rotateValue = (random() - 0.5) * 1.4; // от -0.7 до 0.7

            // Прицеливаемся с большей вероятностью в углы и края карты,
            // где часто прячутся другие танки
            if (random() < 0.4) {
                // Выбираем один из углов или краёв
                const edgeX = random() < 0.5 ? -0.9 : 0.9;
                const edgeY = random() < 0.5 ? -0.9 : 0.9;

                // С небольшим случайным отклонением
                aimX = edgeX + (random() * 0.2 - 0.1);
                aimY = edgeY + (random() * 0.2 - 0.1);
            } else {
                // Просто случайное направление прицела
                aimX = random() * 2 - 1;
                aimY = random() * 2 - 1;
            }

            // Стреляем в случайные моменты
            const shootValue = random() < 0.4 ? 1 : 0;

            return [shootValue, moveValue, rotateValue, aimX, aimY];
        }
    }

    // Запасной вариант - полностью случайные действия
    return generatePureRandomAction();
}

function findNearestEnemy(tankId: number): { id: number; x: number; y: number; dist: number } | null {
    const tankX = TankInputTensor.x[tankId];
    const tankY = TankInputTensor.y[tankId];

    let closestEnemy = null;
    let minDist = Number.MAX_VALUE;

    // Перебираем всех врагов
    for (let j = 0; j < TANK_INPUT_TENSOR_MAX_ENEMIES; j++) {
        const enemyId = TankInputTensor.enemiesData.get(tankId, j * 5);
        const enemyX = TankInputTensor.enemiesData.get(tankId, j * 5 + 1);
        const enemyY = TankInputTensor.enemiesData.get(tankId, j * 5 + 2);

        if (enemyId === 0) continue;

        const dist = Math.hypot(tankX - enemyX, tankY - enemyY);

        if (dist < minDist) {
            minDist = dist;
            closestEnemy = { id: enemyId, x: enemyX, y: enemyY, dist };
        }
    }

    return closestEnemy;
}

function findNearestBullet(tankId: number): {
    id: number;
    x: number;
    y: number;
    vx: number;
    vy: number;
    dist: number;
    danger: number;
} | null {
    const tankX = TankInputTensor.x[tankId];
    const tankY = TankInputTensor.y[tankId];

    let closestBullet = null;
    let maxDanger = 0;

    // Перебираем все пули
    for (let i = 0; i < TANK_INPUT_TENSOR_MAX_BULLETS; i++) {
        const bulletId = TankInputTensor.bulletsData.get(tankId, i * 5);
        const bulletX = TankInputTensor.bulletsData.get(tankId, i * 5 + 1);
        const bulletY = TankInputTensor.bulletsData.get(tankId, i * 5 + 2);
        const bulletVx = TankInputTensor.bulletsData.get(tankId, i * 5 + 3);
        const bulletVy = TankInputTensor.bulletsData.get(tankId, i * 5 + 4);

        if (bulletId === 0) continue;

        // Анализируем опасность пули
        const bulletSpeed = Math.hypot(bulletVx, bulletVy);
        if (bulletSpeed < 0.001) continue;

        const bulletDirX = bulletVx / bulletSpeed;
        const bulletDirY = bulletVy / bulletSpeed;

        // Вектор от пули к танку
        const toTankX = tankX - bulletX;
        const toTankY = tankY - bulletY;

        // Определяем, движется ли пуля к танку
        const dotProduct = toTankX * bulletDirX + toTankY * bulletDirY;

        // Если пуля движется к танку
        if (dotProduct > 0) {
            // Проекция вектора на направление пули
            const projLength = dotProduct;

            // Точка ближайшего прохождения пули к танку
            const closestPointX = bulletX + bulletDirX * projLength;
            const closestPointY = bulletY + bulletDirY * projLength;

            // Расстояние в точке наибольшего сближения
            const minDist = Math.hypot(closestPointX - tankX, closestPointY - tankY);

            // Оценка опасности пули
            if (minDist < 120) { // Увеличенное расстояние обнаружения
                // Время до точки сближения
                const timeToClosest = projLength / bulletSpeed;

                // Плавная оценка опасности
                if (timeToClosest < 1.2) {
                    // Используем smoothstep для плавного изменения опасности
                    const distanceFactor = smoothstep(120, 40, minDist); // От 0 до 1 при приближении
                    const timeFactor = smoothstep(1.2, 0.1, timeToClosest); // От 0 до 1 при приближении

                    const danger = distanceFactor * timeFactor;

                    // Выбираем пулю с наибольшей опасностью
                    if (danger > maxDanger) {
                        maxDanger = danger;
                        closestBullet = {
                            id: bulletId,
                            x: bulletX,
                            y: bulletY,
                            vx: bulletVx,
                            vy: bulletVy,
                            dist: Math.hypot(tankX - bulletX, tankY - bulletY),
                            danger,
                        };
                    }
                }
            }
        }
    }

    return closestBullet;
}

/**
 * Генерация полностью случайных действий
 * @returns Массив с действием [shoot, move, rotate, aimX, aimY]
 */
function generatePureRandomAction(): number[] {
    // Стрельба: случайное 0 или 1
    const shootRandom = random() > 0.5 ? 1 : 0;
    // Движение вперед-назад
    const moveRandom = randomSign() * random();
    // Поворот влево-вправо
    const rotateRandom = randomSign() * random();
    // Прицеливание
    const aimXRandom = randomSign() * random();
    const aimYRandom = randomSign() * random();

    return [shootRandom, moveRandom, rotateRandom, aimXRandom, aimYRandom];
}
