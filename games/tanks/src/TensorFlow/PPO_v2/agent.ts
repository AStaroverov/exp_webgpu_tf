import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM, INPUT_DIM } from '../Common/consts';
import { getCurrentExperiment, RLExperimentConfig } from './experiment-config';
import { Memory } from './Memory.ts';
import { abs, floor } from '../../../../../lib/math.ts';
import { isDevtoolsOpen } from '../Common/utils.ts';

// Общий PPO агент для всех танков
export class SharedTankPPOAgent {
    private iteration = 0;
    private memory: Memory;
    private valueNetwork: tf.LayersModel;   // Сеть критика
    private policyNetwork: tf.LayersModel;  // Сеть политики
    private policyOptimizer!: tf.Optimizer;        // Оптимизатор для policy network
    private valueOptimizer!: tf.Optimizer;         // Оптимизатор для value network
    private config!: RLExperimentConfig;

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
        this.applyConfig(getCurrentExperiment());

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
        this.valueNetwork = this.createValueNetwork();
        this.policyNetwork = this.createPolicyNetwork();

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
    rememberAction(tankId: number, state: tf.Tensor, action: tf.Tensor, logProb: tf.Tensor, value: tf.Tensor) {
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
        for (let i = 0; i < epochs; i++) {
            // Обучение политики
            const policyLoss = this.trainPolicyNetwork(
                batch.states, batch.actions, batch.logProbs, batch.advantages,
            );
            policyLossSum += policyLoss;

            // Обучение критика
            const valueLoss = this.trainValueNetwork(
                batch.states, batch.returns, batch.values,
            );
            valueLossSum += valueLoss;

            console.log(`[Train]: Epoch: ${ i }, Policy loss: ${ policyLoss.toFixed(4) }, Value loss: ${ valueLoss.toFixed(4) }`);
        }
        const newWeights = isDevtoolsOpen() ? this.policyNetwork.getWeights().map(w => w.dataSync()) as Float32Array[] : null;

        isDevtoolsOpen() && console.log('>> WEIGHTS SUM ABS DELTA', newWeights!.reduce((acc, w, i) => {
            return acc + abs(w.reduce((a, b, j) => a + abs(b - prevWeights![i][j]), 0));
        }, 0));

        for (const tensor of Object.values(batch)) {
            if (tensor instanceof tf.Tensor) {
                tensor.dispose();
            }
        }

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
            await this.valueNetwork.save('indexeddb://tank-rl-value-model');
            await this.policyNetwork.save('indexeddb://tank-rl-policy-model');

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
            this.valueNetwork = await tf.loadLayersModel('indexeddb://tank-rl-value-model');
            this.policyNetwork = await tf.loadLayersModel('indexeddb://tank-rl-policy-model');

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
        action: Float32Array,
        logProb: tf.Tensor,
        value: tf.Tensor
    } {
        return tf.tidy(() => {
            const stateTensor = tf.tensor1d(state).expandDims(0);
            const predict = this.policyNetwork.predict(stateTensor) as tf.Tensor;
            const rawOutputSqueezed = predict.squeeze(); // [ACTION_DIM * 2] при batch=1
            const outMean = rawOutputSqueezed.slice([0], [ACTION_DIM]);   // ACTION_DIM штук
            const outLogStd = rawOutputSqueezed.slice([ACTION_DIM], [ACTION_DIM]);
            const clippedLogStd = outLogStd.clipByValue(-2, 0.5);
            const std = clippedLogStd.exp();
            const noise = tf.randomNormal([ACTION_DIM]).mul(std).div(10);
            const action = outMean.add(noise);
            const logProb = this.computeLogProb(outMean, action, std);
            const actionArray = action.dataSync() as Float32Array;
            const value = this.valueNetwork.predict(stateTensor) as tf.Tensor;

            return {
                action: actionArray,
                logProb: logProb,
                value: value.squeeze(),
            };
        });
    }

    // Обучение сети политики
    private trainPolicyNetwork(
        states: tf.Tensor,       // [batchSize, inputDim]
        actions: tf.Tensor,      // [batchSize, actionDim]
        oldLogProbs: tf.Tensor,  // [batchSize] или [batchSize,1]
        advantages: tf.Tensor,   // [batchSize]
    ): number {
        return tf.tidy(() => {
            const totalLoss = this.policyOptimizer.minimize(() => {
                const predict = this.policyNetwork.predict(states) as tf.Tensor;
                const outMean = predict.slice([0, 0], [-1, ACTION_DIM]);
                const outLogStd = predict.slice([0, ACTION_DIM], [-1, ACTION_DIM]);
                const clippedLogStd = outLogStd.clipByValue(-2, 0.5);
                const std = clippedLogStd.exp();
                const newLogProbs = this.computeLogProb(outMean, actions, std);
                const oldLogProbs2D = oldLogProbs.reshape(newLogProbs.shape);
                const ratio = tf.exp(newLogProbs.sub(oldLogProbs2D));
                isDevtoolsOpen() && console.log('>> RATIO SUM ABS DELTA', (ratio.dataSync() as Float32Array).reduce((a, b) => a + abs(1 - b), 0));

                const surr1 = ratio.mul(advantages);
                const clippedRatio = ratio.clipByValue(1 - this.config.clipRatioPolicy, 1 + this.config.clipRatioPolicy);
                const surr2 = clippedRatio.mul(advantages);
                const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1);

                const c = 0.5 * Math.log(2 * Math.PI * Math.E);
                const entropyEachDim = clippedLogStd.add(c); // [batchSize,ACTION_DIM]
                const totalEntropy = entropyEachDim.sum(1).mean();
                const totalLoss = policyLoss.sub(totalEntropy.mul(this.config.entropyCoeff));

                return totalLoss as tf.Scalar;
            }, true);

            if (totalLoss == null) {
                throw new Error('Policy loss is null');
            }

            // Возвращаем число
            return totalLoss!.dataSync()[0];
        });
    }

    // Обучение сети критика (оценка состояний)
    private trainValueNetwork(
        states: tf.Tensor,   // [batchSize, inputDim]
        returns: tf.Tensor,  // [batchSize], уже подсчитанные (GAE + V(s) или просто discountedReturns)
        oldValues: tf.Tensor, // [batchSize], для клиппинга
    ): number {
        return tf.tidy(() => {
            const vfLoss = this.valueOptimizer.minimize(() => {
                // forward pass
                const predicted = this.valueNetwork.predict(states) as tf.Tensor;
                // shape [batchSize,1], приводим к [batchSize]
                const valuePred = predicted.squeeze(); // [batchSize]

                // Клипаем (PPO2 style)
                const oldVal2D = oldValues.reshape(valuePred.shape);   // тоже [batchSize]
                const valuePredClipped = oldVal2D.add(
                    valuePred.sub(oldVal2D).clipByValue(-this.config.clipRatioValue, this.config.clipRatioValue),
                );
                const returns2D = returns.reshape(valuePred.shape);

                const vfLoss1 = returns2D.sub(valuePred).square();
                const vfLoss2 = returns2D.sub(valuePredClipped).square();
                const finalValueLoss = tf.maximum(vfLoss1, vfLoss2).mean().mul(0.5);

                return finalValueLoss as tf.Scalar;
            }, true);

            if (vfLoss == null) {
                throw new Error('Value loss is null');
            }

            return vfLoss!.dataSync()[0];
        });
    }

    private computeLogProb(predict: tf.Tensor, actions: tf.Tensor, scale: tf.Tensor): tf.Tensor {
        return tf.tidy(() => {
            const logUnnormalized = tf.mul(
                -0.5,
                tf.square(
                    tf.sub(
                        tf.div(actions, scale),
                        tf.div(predict, scale),
                    ),
                ),
            );
            const logNormalization = tf.add(
                tf.scalar(0.5 * Math.log(2.0 * Math.PI)),
                tf.log(scale),
            );
            return tf.sum(
                tf.sub(logUnnormalized, logNormalization),
                logUnnormalized.shape.length - 1,
            );
        });
    }

    private applyConfig(config: RLExperimentConfig) {
        this.config = config;
        // TODO: useless on load, fix it later
        this.policyOptimizer = tf.train.adam(this.config.learningRatePolicy);
        this.valueOptimizer = tf.train.adam(this.config.learningRateValue);
    }


    // Создание сети политики
    private createPolicyNetwork(): tf.LayersModel {
        // Входной тензор
        const input = tf.layers.input({ shape: [INPUT_DIM] });

        let x = input;
        for (const [activation, units] of this.config.hiddenLayers) {
            x = tf.layers.dense({
                units,
                activation,
                kernelInitializer: 'glorotUniform',
            }).apply(x) as tf.SymbolicTensor;
        }

        // Выход: ACTION_DIM * 2 нейронов (ACTION_DIM для mean, ACTION_DIM для logStd).
        // При использовании:
        //   mean = tanh(первые ACTION_DIM),
        //   std  = exp(последние ACTION_DIM).
        const policyOutput = tf.layers.dense({
            units: ACTION_DIM * 2,
            activation: 'linear', // без ограничений, трансформации — вручную (tanh/exp)
            name: 'policy_output',
        }).apply(x) as tf.SymbolicTensor;

        // Создаём модель
        return tf.model({
            inputs: input,
            outputs: policyOutput,
        });
    }

    // Создание сети критика (оценки состояний)
    private createValueNetwork(): tf.LayersModel {
        // Входной слой
        const input = tf.layers.input({ shape: [INPUT_DIM] });

        // Скрытые слои
        let x = input;
        for (const [activation, units] of this.config.hiddenLayers) {
            x = tf.layers.dense({
                units,
                activation,
                kernelInitializer: 'glorotUniform',
            }).apply(x) as tf.SymbolicTensor;
        }

        // Выходной слой - скалярная оценка состояния
        const valueOutput = tf.layers.dense({
            units: 1,
            activation: 'linear',
            name: 'value_output',
        }).apply(x) as tf.SymbolicTensor;

        // Создаем модель
        const valueModel = tf.model({
            inputs: input,
            outputs: valueOutput,
        });

        return valueModel;
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
