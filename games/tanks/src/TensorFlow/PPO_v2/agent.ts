import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM, INPUT_DIM } from '../Common/consts';
import { getCurrentExperiment, RLExperimentConfig } from './experiment-config';
import { random } from '../../../../../lib/random.ts';
import { generateGuidedRandomAction, generatePureRandomAction } from './generateActions.ts';

// Буфер опыта для PPO
class PPOMemory {
    private states: tf.Tensor[] = [];
    private actions: tf.Tensor[] = [];
    private logProbs: tf.Tensor[] = [];
    private values: tf.Tensor[] = [];
    private rewards: number[] = [];
    private dones: boolean[] = [];

    constructor() {
    }


    size() {
        return this.states.length;
    }

    add(state: tf.Tensor, action: tf.Tensor, logProb: tf.Tensor, value: tf.Tensor, reward: number, done: boolean) {
        this.states.push(state.clone());
        this.actions.push(action.clone());
        this.logProbs.push(logProb.clone());
        this.values.push(value.clone());
        this.rewards.push(reward);
        this.dones.push(done);
    }

    // Метод для получения батча для обучения
    getBatch() {
        if (this.states.length === 0) {
            throw new Error('Memory is empty');
        }

        return {
            states: tf.stack(this.states),
            actions: tf.stack(this.actions),
            logProbs: tf.stack(this.logProbs),
            values: tf.stack(this.values),
            rewards: tf.tensor1d(this.rewards),
            dones: tf.tensor1d(this.dones.map(done => done ? 1.0 : 0.0)),
        };
    }

    computeReturnsAndAdvantages(gamma: number, lam: number, lastValue: number = 0) {
        const n = this.states.length;
        const returns: number[] = new Array(n).fill(0);
        const advantages: number[] = new Array(n).fill(0);

        // Скачаем values в CPU, чтоб проще считать
        const valuesArr = tf.stack(this.values).dataSync(); // shape [n]

        let adv = 0;
        // bootstrap, если последний transition не done
        let lastVal = lastValue; // Если done в конце, возьмём 0

        // Идём с конца вперёд
        for (let i = n - 1; i >= 0; i--) {
            if (this.dones[i]) {
                // если done, то обнуляем хвост
                adv = 0;
                lastVal = 0;
            }
            const delta = this.rewards[i]
                + gamma * lastVal * (this.dones[i] ? 0 : 1)
                - valuesArr[i];
            adv = delta + gamma * lam * adv * (this.dones[i] ? 0 : 1);

            advantages[i] = adv;
            returns[i] = valuesArr[i] + adv;

            lastVal = valuesArr[i];
        }

        return { returns: tf.tensor1d(returns), advantages: tf.tensor1d(advantages) };
    }

    dispose() {
        // Освобождаем все тензоры
        this.states.forEach(state => state.dispose());
        this.actions.forEach(action => action.dispose());
        this.logProbs.forEach(logProb => logProb.dispose());
        this.values.forEach(value => value.dispose());

        // Сбрасываем массивы
        this.states = [];
        this.actions = [];
        this.logProbs = [];
        this.values = [];
        this.rewards = [];
        this.dones = [];
    }

}

// Общий PPO агент для всех танков
export class SharedTankPPOAgent {
    private memory: PPOMemory;
    private policyNetwork: tf.LayersModel;  // Сеть политики
    private valueNetwork: tf.LayersModel;   // Сеть критика
    private optimizer: tf.Optimizer;        // Оптимизатор для обучения
    private epsilon: number;                // Параметр для исследования среды
    private config: RLExperimentConfig;
    private ppoEpsilon: number;             // Параметр клиппирования для PPO
    private epochs: number;                 // Количество эпох обучения на одном батче
    private entropyCoeff: number;           // Коэффициент энтропии для поощрения исследования

    private logger: {
        episodeRewards: number[];
        episodeLengths: number[];
        epsilon: number[];
        losses: {
            policy: number[];
            value: number[];
        };
    };

    constructor() {
        // Получаем текущую конфигурацию эксперимента
        this.config = getCurrentExperiment();

        // Инициализируем с значениями из конфига
        this.memory = new PPOMemory();
        this.epsilon = this.config.epsilon;
        this.optimizer = tf.train.adam(this.config.learningRate);

        // PPO-специфичные параметры
        this.ppoEpsilon = 0.2;              // Клиппирование соотношения вероятностей
        this.epochs = 4;                    // Количество эпох обучения на одном батче
        this.entropyCoeff = 0.01;           // Коэффициент энтропии

        // Инициализируем логгер
        this.logger = {
            episodeRewards: [],
            episodeLengths: [],
            epsilon: [],
            losses: {
                policy: [],
                value: [],
            },
        };

        // Создаем модели
        this.policyNetwork = this.createPolicyNetwork();
        this.valueNetwork = this.createValueNetwork();

        console.log(`Shared PPO Agent initialized with experiment: ${ this.config.name }`);
    }

    // Метод для сохранения опыта в буфер
    remember(state: tf.Tensor, action: tf.Tensor, logProb: tf.Tensor, value: tf.Tensor, reward: number, done: boolean) {
        this.memory.add(state, action, logProb, value, reward, done);
    }

    /**
     * Выбор действия с использованием политики
     * @param state Состояние танка
     * @param tankId ID танка
     * @param isTraining Находимся ли мы в режиме обучения
     * @returns Объект с действием, логарифмом вероятности и значением состояния
     */
    act(state: tf.Tensor, tankId: number, isTraining: boolean = true): {
        onPolicy: true,
        action: number[],
        logProb: tf.Tensor,
        value: tf.Tensor
    } | {
        onPolicy: false,
        action: number[],
    } {
        // Если не в режиме обучения или случайное значение больше epsilon, используем сеть
        if (!isTraining || random() > this.epsilon) {
            return this.getActionFromPolicy(state);
        }

        // В противном случае, используем стратегию исследования
        const strategy = selectStrategy(this.epsilon, this.config.epsilon, this.config.epsilonMin);

        let action: number[];
        switch (strategy) {
            case 'pure_random':
                // Полностью случайные действия
                action = generatePureRandomAction();
                break;
            case 'guided_random':
                // Управляемое случайное исследование
                action = generateGuidedRandomAction(tankId);
                break;
            default:
                // Используем обученную модель
                return this.getActionFromPolicy(state);
        }

        return { onPolicy: false, action };
    }

    // Обучение агента по собранному опыту
    train() {
        const batch = this.memory.getBatch();

        const { states, actions, logProbs: oldLogProbs, values: oldValues, rewards, dones } = batch;
        const { returns, advantages } = this.memory.computeReturnsAndAdvantages(
            this.config.gamma,
            this.config.lam,    // lambda GAE
            0,                   // lastValue если эпизод не done
        );

        let policyLossSum = 0, valueLossSum = 0;
        // 2) Несколько эпох PPO
        for (let epoch = 0; epoch < this.epochs; epoch++) {
            // Обучение политики
            const policyLoss = this.trainPolicyNetwork(
                states, actions, oldLogProbs, advantages,
            );
            policyLossSum += policyLoss;

            // Обучение критика
            const valueLoss = this.trainValueNetwork(
                states, returns, oldValues,
            );
            valueLossSum += valueLoss;

        }

        states.dispose();
        actions.dispose();
        oldLogProbs.dispose();
        oldValues.dispose();
        rewards.dispose();
        dones.dispose();
        returns.dispose();
        advantages.dispose();

        const avgPolicyLoss = policyLossSum / this.epochs;
        const avgValueLoss = valueLossSum / this.epochs;
        // Лог
        this.logger.losses.policy.push(avgPolicyLoss);
        this.logger.losses.value.push(avgValueLoss);

        // По желанию — очистить память (если вы обучаетесь по эпизодам)
        this.memory.dispose();

        return { policy: avgPolicyLoss, value: avgValueLoss };
    }

    // Обновление epsilon для баланса исследования/эксплуатации
    updateEpsilon() {
        if (this.epsilon > this.config.epsilonMin) {
            this.epsilon *= this.config.epsilonDecay;
        }
        this.logger.epsilon.push(this.epsilon);
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
            epsilon: this.epsilon,
            memorySize: this.memory.size(),
            avgReward: avgReward,
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
                epsilon: this.epsilon,
                experimentName: this.config.name,
                timestamp: new Date().toISOString(),
                losses: {
                    policy: this.logger.losses.policy.slice(-10),
                    value: this.logger.losses.value.slice(-10),
                },
                rewards: this.logger.episodeRewards.slice(-10), // Последние 10 наград
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
            this.policyNetwork = await tf.loadLayersModel('indexeddb://tank-rl-policy-model');
            this.valueNetwork = await tf.loadLayersModel('indexeddb://tank-rl-value-model');

            console.log('PPO models loaded successfully');

            const metadata = JSON.parse(localStorage.getItem('tank-rl-agent-state') ?? '{}');
            if (metadata) {
                this.epsilon = metadata.epsilon || this.config.epsilon;

                // Загружаем историю потерь и наград, если она есть
                if (metadata.losses) {
                    this.logger.losses = metadata.losses;
                }

                if (metadata.rewards && metadata.rewards.length > 0) {
                    this.logger.episodeRewards = metadata.rewards;
                }

                console.log(`Restored training state epsilon ${ this.epsilon.toFixed(4) }`);

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

    // Освобождение ресурсов
    dispose() {
        this.policyNetwork.dispose();
        this.valueNetwork.dispose();
        this.memory.dispose();
        console.log('PPO Agent resources disposed');
    }

    // Получение действия из сети политики
    private getActionFromPolicy(state: tf.Tensor): {
        onPolicy: boolean,
        action: number[],
        logProb: tf.Tensor,
        value: tf.Tensor
    } {
        return tf.tidy(() => {
            const stateTensor = state.expandDims(0);

            // Получаем выходы из сети политики
            const policyOutputs = this.policyNetwork.predict(stateTensor) as tf.Tensor[];

            // Параметры распределений для различных компонентов действия

            // 1. Стрельба (дискретное действие с распределением Бернулли)
            const shootProb = policyOutputs[0].squeeze();

            // 2. Движение (непрерывное действие с нормальным распределением)
            const moveMean = policyOutputs[1].squeeze();
            const moveStd = tf.exp(policyOutputs[2].squeeze()).clipByValue(0.1, 1.0);

            // 3. Прицеливание (непрерывное действие с нормальным распределением)
            const aimMean = policyOutputs[3].squeeze();
            const aimStd = tf.exp(policyOutputs[4].squeeze()).clipByValue(0.1, 1.0);

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

            // Комбинируем в одно действие
            const action = [
                ...shootAction.dataSync(),
                ...moveAction.dataSync(),
                ...aimAction.dataSync(),
            ];

            if (action.length !== ACTION_DIM) {
                throw new Error(`Invalid action length: ${ action.length }`);
            }

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

            return {
                action,
                onPolicy: true,
                logProb: totalLogProb,
                value: value.squeeze(),
            };
        });
    }

    // Обучение сети политики
    private trainPolicyNetwork(
        states: tf.Tensor,       // [batchSize, inputDim]
        actions: tf.Tensor,      // [batchSize, 5] (shoot=1, move=2, aim=2)
        oldLogProbs: tf.Tensor,  // [batchSize] или [batchSize,1]
        advantages: tf.Tensor,   // [batchSize]
    ): number {
        // 1) Собираем массив tf.Variable
        const trainableVars = this.policyNetwork.trainableWeights.map(
            // @ts-ignore
            w => w.val,
        );

        // 2) Всё оборачиваем в tf.tidy
        //    и всю логику вычисления лосса — внутрь колбэка variableGrads
        return tf.tidy(() => {
            const { value: totalLoss, grads } = tf.variableGrads(() => {
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
                const clippedMoveLogStd = moveLogStd.clipByValue(-2, 2);
                const moveStd = clippedMoveLogStd.exp();
                const moveDiff = moveActions.sub(moveMean);
                // logprob по оси=2
                const moveLogProbEachDim = moveDiff.square().div(moveStd.square().add(1e-8))
                    .mul(-0.5)
                    .sub(tf.log(moveStd.mul(Math.sqrt(2 * Math.PI))));
                const moveLogProb = moveLogProbEachDim.sum(1, true); // [batchSize,1]

                // -- aim (Normal, 2D)
                const clippedAimLogStd = aimLogStd.clipByValue(-2, 2);
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

                // g) считаем surrogate1, surrogate2
                const surr1 = ratio.mul(advantages);
                const clippedRatio = ratio.clipByValue(1 - this.ppoEpsilon, 1 + this.ppoEpsilon);
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
                const totalLoss = policyLoss.sub(totalEntropy.mul(this.entropyCoeff));

                return totalLoss as tf.Scalar;
            }, trainableVars);

            // j) Применяем градиенты
            this.optimizer.applyGradients(grads);

            // Возвращаем число
            return totalLoss.dataSync()[0];
        });
    }

    // Обучение сети критика (оценка состояний)
    private trainValueNetwork(
        states: tf.Tensor,   // [batchSize, inputDim]
        returns: tf.Tensor,  // [batchSize], уже подсчитанные (GAE + V(s) или просто discountedReturns)
        oldValues: tf.Tensor, // [batchSize], для клиппинга
    ): number {
        // Собираем tf.Variable
        const trainableVars = this.valueNetwork.trainableWeights.map(
            // @ts-ignore
            w => w.val,
        );

        return tf.tidy(() => {
            const { value: vfLoss, grads } = tf.variableGrads(() => {
                // forward pass
                const predicted = this.valueNetwork.apply(states, { training: true }) as tf.Tensor;
                // shape [batchSize,1], приводим к [batchSize]
                const valuePred = predicted.squeeze(); // [batchSize]

                // Клипаем (PPO2 style)
                const oldVal2D = oldValues.reshape(valuePred.shape);   // тоже [batchSize]
                const valuePredClipped = oldVal2D.add(
                    valuePred.sub(oldVal2D).clipByValue(-this.ppoEpsilon, this.ppoEpsilon),
                );
                const returns2D = returns.reshape(valuePred.shape);

                const vfLoss1 = returns2D.sub(valuePred).square();
                const vfLoss2 = returns2D.sub(valuePredClipped).square();
                const finalValueLoss = tf.maximum(vfLoss1, vfLoss2).mean();

                return finalValueLoss as tf.Scalar;
            }, trainableVars);

            this.optimizer.applyGradients(grads);

            return vfLoss.dataSync()[0];
        });
    }

    // Создание сети политики
    private createPolicyNetwork(): tf.LayersModel {
        // Входной слой
        const input = tf.layers.input({ shape: [INPUT_DIM] });

        // Общие слои
        let shared = input;
        for (const units of this.config.hiddenLayers) {
            shared = tf.layers.dense({
                units,
                activation: 'relu',
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
            activation: 'tanh',
            name: 'move_log_std',
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
            activation: 'tanh',
            name: 'aim_log_std',
        }).apply(shared) as tf.SymbolicTensor;

        // Создаем модель с множественными выходами
        const policyModel = tf.model({
            inputs: input,
            outputs: [shootProb, moveMean, moveLogStd, aimMean, aimLogStd],
        });

        // Компилируем модель
        policyModel.compile({
            optimizer: this.optimizer,
            loss: {
                shoot_prob: 'binaryCrossentropy',
                move_mean: 'meanSquaredError',
                move_log_std: 'meanSquaredError',
                aim_mean: 'meanSquaredError',
                aim_log_std: 'meanSquaredError',
            },
        });

        return policyModel;
    }

    // Создание сети критика (оценки состояний)
    private createValueNetwork(): tf.LayersModel {
        // Входной слой
        const input = tf.layers.input({ shape: [INPUT_DIM] });

        // Скрытые слои
        let x = input;
        for (const units of this.config.hiddenLayers) {
            x = tf.layers.dense({
                units,
                activation: 'relu',
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

        // Компилируем модель
        valueModel.compile({
            optimizer: this.optimizer,
            loss: 'meanSquaredError',
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

/**
 * Выбор стратегии на основе текущего значения epsilon и других факторов
 * @returns Выбранная стратегия
 */
function selectStrategy(epsilon: number, epsilonBase: number, epsilonMin: number): 'pure_random' | 'guided_random' | 'exploitation' {
    // По мере уменьшения epsilon, мы снижаем вероятность случайных действий
    // и увеличиваем вероятность использования модели

    // Начальное распределение:
    // - 5% полностью случайные
    // - 10% управляемые случайные

    // Конечное распределение:
    // - 1% полностью случайные
    // - 1% управляемые случайные

    // Вычисляем, как далеко мы продвинулись в обучении (от 0 до 1)
    const progress = 1 - ((epsilon - epsilonMin) /
        (epsilonBase - epsilonMin));

    // Вычисляем вероятности каждой стратегии
    const pureRandomProb = 0.05 - 0.04 * progress;
    const guidedRandomProb = 0.1 - 0.09 * progress;
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