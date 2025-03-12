import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM, INPUT_DIM } from '../Common/consts';
import { getCurrentExperiment, RLExperimentConfig } from './experiment-config';
import { Memory } from './Memory.ts';
import { Actions, createAction } from './utils.ts';
import { abs, floor } from '../../../../../lib/math.ts';
import { isDevtoolsOpen } from '../Common/utils.ts';

// Общий PPO агент для всех танков
export class SharedTankPPOAgent {
    private iteration = 0;
    private memory: Memory;
    private valueNetwork: tf.LayersModel;   // Сеть критика
    private policyNetwork: tf.LayersModel;  // Сеть политики
    private policyNetworkOld!: tf.LayersModel;
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
        this.policyNetworkOld = this.createPolicyNetwork();
        this.policyNetworkOld.setWeights(
            this.policyNetwork.getWeights().map(w => w.clone()),
        );

        console.log(`Shared PPO Agent initialized with experiment: ${ this.config.name }`);
    }

    // Освобождение ресурсов
    dispose() {
        this.policyNetwork.dispose();
        this.policyNetworkOld.dispose();
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

    act(state: Float32Array): {
        action: Actions,
        logProb: tf.Tensor,
        value: tf.Tensor
    } {
        return tf.tidy(() => {
            const stateTensor = tf.tensor1d(state).expandDims(0);

            // Получаем выходы из сети политики
            const policyOutputs = this.policyNetworkOld.predict(stateTensor) as tf.Tensor[];

            // Параметры распределений для различных компонентов действия

            // 1. Стрельба (дискретное действие с распределением Бернулли)
            const shootProb = policyOutputs[0].squeeze();

            // 2. Движение (непрерывное действие с нормальным распределением)
            const moveMean = policyOutputs[1].squeeze();
            const moveStd = policyOutputs[2].squeeze().clipByValue(-5, 0.5).exp();

            // 3. Прицеливание (непрерывное действие с нормальным распределением)
            const aimMean = policyOutputs[3].squeeze();
            const aimStd = policyOutputs[4].squeeze().clipByValue(-5, 0.5).exp();

            // Сэмплируем действия из соответствующих распределений

            // Стрельба (бинарное действие)
            const shootRandom = tf.less(tf.randomUniform([1]), shootProb);
            const shootAction = shootRandom.asType('float32');

            // Движение (непрерывное действие)
            const moveNoise = tf.randomNormal([2]);
            const moveAction = moveMean.add(moveNoise.mul(moveStd));

            // Прицеливание (непрерывное действие)
            const aimNoise = tf.randomNormal([2]);
            const aimAction = aimMean.add(aimNoise.mul(aimStd));

            // Вычисляем логарифм вероятности действия

            // Логарифм вероятности стрельбы (распределение Бернулли)
            const shootLogProb = tf.where(
                tf.equal(shootAction, 1),
                tf.log(shootProb.add(1e-8)),
                tf.log(tf.scalar(1).sub(shootProb).add(1e-8)),
            );

            // Логарифм вероятности движения (нормальное распределение)
            const moveLogProbDist = tf.sub(moveAction, moveMean).div(moveStd.add(1e-8)).square().mul(-0.5)
                .sub(tf.log(moveStd.mul(Math.sqrt(2 * Math.PI))));

            // Логарифм вероятности прицеливания (нормальное распределение)
            const aimLogProbDist = tf.sub(aimAction, aimMean).div(aimStd.add(1e-8)).square().mul(-0.5)
                .sub(tf.log(aimStd.mul(Math.sqrt(2 * Math.PI))));

            // Объединяем логарифмы вероятностей (сумма, т.к. компоненты независимы)
            let totalLogProb = shootLogProb.add(moveLogProbDist.sum()).add(aimLogProbDist.sum());

            totalLogProb = totalLogProb.squeeze();

            // Получаем оценку состояния из сети критика
            const value = this.valueNetwork.predict(stateTensor) as tf.Tensor;

            // Комбинируем в одно действие
            const action = createAction(
                shootAction.dataSync()[0],
                moveAction.dataSync() as Float32Array,
                aimAction.dataSync() as Float32Array,
            );

            if (action.length !== ACTION_DIM) {
                throw new Error(`Invalid action length: ${ action.length }`);
            }

            return {
                action,
                logProb: totalLogProb,
                value: value.squeeze(),
            };
        });
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

        this.policyNetworkOld.setWeights(
            this.policyNetwork.getWeights().map(w => w.clone()),
        );

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
            await this.policyNetworkOld.save('indexeddb://tank-rl-policy-old-model');

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

    // Загрузка модели из хранилища
    async load() {
        try {
            this.valueNetwork = await tf.loadLayersModel('indexeddb://tank-rl-value-model');
            this.policyNetwork = await tf.loadLayersModel('indexeddb://tank-rl-policy-model');
            this.policyNetworkOld = await tf.loadLayersModel('indexeddb://tank-rl-policy-old-model');

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

    private applyConfig(config: RLExperimentConfig) {
        this.config = config;
        // TODO: useless on load, fix it later
        this.policyOptimizer = tf.train.adam(this.config.learningRatePolicy);
        this.valueOptimizer = tf.train.adam(this.config.learningRateValue);
    }

    // Обучение сети политики
    private trainPolicyNetwork(
        states: tf.Tensor,       // [batchSize, inputDim]
        actions: tf.Tensor,      // [batchSize, 5] (shoot=1, move=2, aim=2)
        oldLogProbs: tf.Tensor,  // [batchSize] или [batchSize,1]
        advantages: tf.Tensor,   // [batchSize]
    ): number {
        return tf.tidy(() => {
            const totalLoss = this.policyOptimizer.minimize(() => {
                // a) forward pass через policyNetwork в режиме обучения
                //    Важно: НЕ predict(...), а apply(..., {training: true})
                const policyOutputs = this.policyNetwork.apply(states, { training: true }) as tf.Tensor[];
                // policyOutputs = [shootProb, moveMean, moveLogStd, aimMean, aimLogStd]
                const shootProb = policyOutputs[0]; // [batchSize,1]
                const moveMean = policyOutputs[1]; // [batchSize,2]
                const moveLogStd = policyOutputs[2]; // [batchSize,2]
                const aimMean = policyOutputs[3]; // [batchSize,2]
                const aimLogStd = policyOutputs[4]; // [batchSize,2]

                // b) Разбиваем `actions` на shoot / move / aim
                const shootActions = actions.slice([0, 0], [-1, 1]); // [batchSize,1]
                const moveActions = actions.slice([0, 1], [-1, 2]); // [batchSize,2]
                const aimActions = actions.slice([0, 3], [-1, 2]); // [batchSize,2]

                // c) Считаем logProb для каждого компонента (Bernoulli и Normal)
                // -- shoot (Bernoulli)
                const shootLogProb = tf.where(
                    tf.equal(shootActions, 1),
                    tf.log(shootProb.add(1e-8)),
                    tf.log(tf.scalar(1).sub(shootProb).add(1e-8)),
                ); // [batchSize,1]

                // -- move (Normal, 2D)
                const clippedMoveLogStd = moveLogStd.clipByValue(-5, 0.5);
                const moveStd = clippedMoveLogStd.exp();
                const moveDiff = moveActions.sub(moveMean);
                // logprob по оси=2
                const moveLogProbEachDim = moveDiff.square().div(moveStd.square().add(1e-8))
                    .mul(-0.5)
                    .sub(tf.log(moveStd.mul(Math.sqrt(2 * Math.PI))));
                const moveLogProb = moveLogProbEachDim.sum(1, true); // [batchSize,1]

                // -- aim (Normal, 2D)
                const clippedAimLogStd = aimLogStd.clipByValue(-5, 0.5);
                const aimStd = clippedAimLogStd.exp();
                const aimDiff = aimActions.sub(aimMean);
                const aimLogProbEachDim = aimDiff.square().div(aimStd.square().add(1e-8))
                    .mul(-0.5)
                    .sub(tf.log(aimStd.mul(Math.sqrt(2 * Math.PI))));
                const aimLogProb = aimLogProbEachDim.sum(1, true); // [batchSize,1]

                // d) Итоговый logProb на всё действие
                const newLogProbs = shootLogProb.add(moveLogProb).add(aimLogProb); // [batchSize,1]

                // e) Приводим oldLogProbs к [batchSize,1] (если нужно)
                const oldLogProbs2D = oldLogProbs.reshape(newLogProbs.shape);

                // f) ratio = exp(newLogProb - oldLogProb)
                const ratio = tf.exp(newLogProbs.sub(oldLogProbs2D));
                isDevtoolsOpen() && console.log('>> RATIO SUM ABS DELTA', (ratio.dataSync() as Float32Array).reduce((a, b) => a + abs(1 - b), 0));

                // g) считаем surrogate1, surrogate2
                const surr1 = ratio.mul(advantages);
                const clippedRatio = ratio.clipByValue(1 - this.config.clipRatio, 1 + this.config.clipRatio);
                const surr2 = clippedRatio.mul(advantages);

                // policyLoss = - mean( min(surr1, surr2) )
                let policyLoss = tf.minimum(surr1, surr2).mean().mul(-1);

                // h) Энтропия
                // -- shootEntropy
                const shootEntropy = shootProb.mul(tf.log(shootProb.add(1e-8)))
                    .add(tf.scalar(1).sub(shootProb).mul(tf.log(tf.scalar(1).sub(shootProb).add(1e-8))))
                    .mul(-1) // [batchSize,1]
                    .mean(); // скаляр

                // -- moveEntropy
                const c = 0.5 * Math.log(2 * Math.PI * Math.E);
                const moveEntropyEach = clippedMoveLogStd.add(c); // [batchSize,2]
                const moveEntropyMean = moveEntropyEach.sum(1).mean(); // скаляр

                // -- aimEntropy
                const aimEntropyEach = clippedAimLogStd.add(c);  // [batchSize,2]
                const aimEntropyMean = aimEntropyEach.sum(1).mean();

                const totalEntropy = shootEntropy.add(moveEntropyMean).add(aimEntropyMean);

                // i) full loss = policyLoss - entropyCoeff * entropy
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
                const predicted = this.valueNetwork.apply(states, { training: true }) as tf.Tensor;
                // shape [batchSize,1], приводим к [batchSize]
                const valuePred = predicted.squeeze(); // [batchSize]

                // Клипаем (PPO2 style)
                const oldVal2D = oldValues.reshape(valuePred.shape);   // тоже [batchSize]
                const valuePredClipped = oldVal2D.add(
                    valuePred.sub(oldVal2D).clipByValue(-this.config.clipRatio, this.config.clipRatio),
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

    // Создание сети политики
    private createPolicyNetwork(): tf.LayersModel {
        // Входной слой
        const input = tf.layers.input({ shape: [INPUT_DIM] });

        // Общие слои
        let shared = input;
        for (const [activation, units] of this.config.hiddenLayers) {
            shared = tf.layers.dense({
                units,
                activation,
                kernelInitializer: 'glorotUniform',
            }).apply(shared) as tf.SymbolicTensor;
        }
        // Выходные головы для политики

        // 1. Стрельба - вероятность стрельбы (Бернулли)
        const shootProb = tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            name: 'shoot_prob',
        }).apply(shared) as tf.SymbolicTensor;

        // 2. Движение - среднее значение (mu) нормального распределения
        const moveMean = tf.layers.dense({
            units: 2,
            activation: 'tanh',  // Выход в диапазоне [-1, 1]
            name: 'move_mean',
        }).apply(shared) as tf.SymbolicTensor;

        // 3. Стандартное отклонение для движения (параметр масштаба)
        const moveLogStd = tf.layers.dense({
            units: 2,
            activation: 'linear',
            name: 'move_log_std',
            kernelInitializer: 'zeros',      // веса в ноль
            biasInitializer: tf.initializers.constant({ value: -1 }),  // логарифм ≈ -1
        }).apply(shared) as tf.SymbolicTensor;

        // 4. Прицеливание - среднее значение (mu) нормального распределения
        const aimMean = tf.layers.dense({
            units: 2,
            activation: 'tanh',  // Выход в диапазоне [-1, 1]
            name: 'aim_mean',
        }).apply(shared) as tf.SymbolicTensor;

        // 5. Стандартное отклонение для прицеливания (параметр масштаба)
        const aimLogStd = tf.layers.dense({
            units: 2,
            activation: 'linear',
            name: 'aim_log_std',
            kernelInitializer: 'zeros',      // веса в ноль
            biasInitializer: tf.initializers.constant({ value: -1 }),  // логарифм ≈ -1
        }).apply(shared) as tf.SymbolicTensor;

        // Создаем модель с множественными выходами
        const policyModel = tf.model({
            inputs: input,
            outputs: [shootProb, moveMean, moveLogStd, aimMean, aimLogStd],
        });

        return policyModel;
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
