import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM, INPUT_DIM } from '../../Common/consts.ts';
import { Memory } from '../Memory.ts';
import { getCurrentExperiment, RLExperimentConfig } from '../config.ts';
import { getAgentState } from '../Database.ts';
import { computeLogProbTanh } from '../../Common/computeLogProb.ts';

export type TensorData = {
    data: Float32Array,
    shape: number[]
}
export type GradientData = { loss: number, grads: { [p: string]: TensorData } };
export type GradientsData = { policy: GradientData; value: GradientData, avgReward: number };

export class SlaveAgent {
    private memory: Memory;
    private config!: RLExperimentConfig;
    private policyNetwork!: tf.LayersModel;
    private policyOptimizer!: tf.Optimizer;
    private valueNetwork!: tf.LayersModel;
    private valueOptimizer!: tf.Optimizer;

    constructor() {
        this.memory = new Memory();
        this.config = getCurrentExperiment();
    }

    public static create() {
        return new SlaveAgent();
    }

    dispose() {
        this.policyNetwork?.dispose();
        this.policyOptimizer?.dispose();
        this.valueNetwork?.dispose();
        this.valueOptimizer?.dispose();
        this.memory.dispose();
    }

    isReady() {
        return this.memory.toArray().some((m) => m.size() >= this.config.batchSize);
    }

    rememberAction(tankId: number, state: Float32Array, action: Float32Array, value: tf.Tensor) {
        this.memory.addFirstPart(tankId, state, action, value);
    }

    rememberReward(tankId: number, reward: number, done: boolean, isLast = false) {
        this.memory.updateSecondPart(tankId, reward, done, isLast);
    }

    async load() {
        try {
            const policyNetwork = await tf.loadLayersModel('indexeddb://tank-rl-policy-model');
            const valueNetwork = await tf.loadLayersModel('indexeddb://tank-rl-value-model');
            const agentState = await getAgentState();

            if (agentState && agentState.config) {
                this.policyNetwork = policyNetwork;
                this.valueNetwork = valueNetwork;
                this.applyConfig(agentState.config);
                return true;
            }

            return false;
        } catch (error) {
            console.warn('Could not load PPO models, starting with new ones:', error);
            return false;
        }
    }

    public act(state: Float32Array): {
        rawActions: Float32Array,
        actions: Float32Array,
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
            const noise = tf.randomNormal([ACTION_DIM]).mul(std);
            const actions = outMean.add(noise);
            const value = this.valueNetwork.predict(stateTensor) as tf.Tensor;

            return {
                rawActions: actions.dataSync() as Float32Array,
                actions: actions.tanh().dataSync() as Float32Array,
                value: value.squeeze(),
            };
        });
    }

    public computeGradients(): GradientsData {
        const batch = this.memory.getBatch(
            this.config.gamma,
            this.config.lam,
        );
        const size = batch.size;
        const tStates = tf.tensor(batch.states, [size, INPUT_DIM]);
        const tActions = tf.tensor(batch.actions, [size, ACTION_DIM]);
        const tAdvantages = tf.tensor(batch.advantages, [size]);
        const tReturns = tf.tensor(batch.returns, [size]);

        const policy = this.computePolicyGradients(tStates, tActions, tAdvantages);
        const value = this.computeValueGradients(tStates, tReturns);

        tStates.dispose();
        tActions.dispose();
        tAdvantages.dispose();
        tReturns.dispose();

        const avgReward = batch.rewards.reduce((a, b) => a + b, 0) / size;

        return { policy, value, avgReward };
    }

    private computePolicyGradients(
        states: tf.Tensor,       // [batchSize, inputDim]
        actions: tf.Tensor,      // [batchSize, actionDim]
        advantages: tf.Tensor,   // [batchSize]
    ): GradientData {
        const { value, grads } = this.policyOptimizer.computeGradients(() => {
            // 1) Прогоняем states через сеть, получаем среднее и log-std (или logits, если дискретные действия)
            const predict = this.policyNetwork.predict(states) as tf.Tensor;
            const outMean = predict.slice([0, 0], [-1, ACTION_DIM]);
            const outLogStd = predict.slice([0, ACTION_DIM], [-1, ACTION_DIM]);
            // 2) Можно «клиповать» log std, чтобы не выходило за границы
            const clippedLogStd = outLogStd.clipByValue(-2, 0.5);
            const std = clippedLogStd.exp();
            // 3) Вычисляем лог-вероятности действий
            const newLogProbs = computeLogProbTanh(actions, outMean, std);
            // 4) Считаем убыток политики: -E[ A * log(pi(a|s)) ]
            //    (т.е. берем среднее от - (advantages * newLogProbs))
            const policyLoss = newLogProbs.mul(advantages).mean().mul(-1);
            // 5) Энтропия (для гауссиан):
            //    Для 1D Гауссианы энтропия = 0.5 * log(2 * π * e) + logStd
            //    Но у нас dims = ACTION_DIM, значит суммируем по каждому действию
            const c = 0.5 * Math.log(2 * Math.PI * Math.E);
            // [batchSize, ACTION_DIM]
            const entropyEachDim = clippedLogStd.add(c);
            // суммируем энтропию по actionDim -> [batchSize], и берём среднее
            const totalEntropy = entropyEachDim.sum(1).mean();
            // 6) Общая функция потерь:
            //    totalLoss = policyLoss - entropyCoeff * entropy
            //    (обычно энтропию вычитают, чтобы увеличить её вклад => «замедляя» переобучение политики)
            const totalLoss = policyLoss.sub(totalEntropy.mul(this.config.entropyCoeff));

            return totalLoss as tf.Scalar;
        });

        const valueNumber = value.dataSync()[0];
        value.dispose();

        const gradsRecord = Object.entries(grads).reduce((acc, [key, tensor]) => {
            acc[removeDigitPostfix(key)] = {
                data: tensor.dataSync() as Float32Array,
                shape: tensor.shape,
            };
            tensor.dispose();
            return acc;
        }, {} as { [name: string]: TensorData });

        return { loss: valueNumber, grads: gradsRecord };
    }

    private computeValueGradients(
        states: tf.Tensor,   // [batchSize, inputDim]
        returns: tf.Tensor,  // [batchSize], уже подсчитанные (GAE + V(s) или просто discountedReturns)
    ): GradientData {
        const { value, grads } = this.valueOptimizer.computeGradients(() => {
            // forward pass
            const predicted = this.valueNetwork.predict(states) as tf.Tensor;
            // [batchSize, 1] -> [batchSize]
            const valuePred = predicted.squeeze();
            // простой MSE (L2) между returns и valuePred
            const returns2D = returns.reshape(valuePred.shape);
            const mse = returns2D.sub(valuePred).square().mean().mul(0.5);

            return mse as tf.Scalar;
        });

        const valueNumber = value.dataSync()[0];
        value.dispose();

        const gradsRecord = Object.entries(grads).reduce((acc, [key, tensor]) => {
            acc[removeDigitPostfix(key)] = {
                data: tensor.dataSync() as Float32Array,
                shape: tensor.shape,
            };
            tensor.dispose();
            return acc;
        }, {} as { [name: string]: TensorData });

        return { loss: valueNumber, grads: gradsRecord };
    }

    private applyConfig(config: RLExperimentConfig) {
        this.config = config;
        this.policyOptimizer = tf.train.adam(this.config.learningRatePolicy);
        this.valueOptimizer = tf.train.adam(this.config.learningRateValue);
    }
}

function removeDigitPostfix(str: string): string {
    const match = str.match(/_(\d+)$/);
    return match !== null ? str.replace(match[0], '') : str;
}