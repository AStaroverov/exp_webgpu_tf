import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { getCurrentExperiment, RLExperimentConfig } from '../config.ts';
import { ACTION_DIM } from '../../Common/consts.ts';
import { GradientData } from '../Slave/SlaveAgent.ts';
import { clearGradientsList, getAgentState, getGradientsList, setAgentState } from '../Database.ts';
import { RingBuffer } from 'ring-buffer-ts';
import { isDevtoolsOpen } from '../../Common/utils.ts';
import { abs } from '../../../../../../lib/math.ts';
import { createPolicyNetwork, createValueNetwork } from '../../Common/models.ts';

export class MasterAgent {
    private config!: RLExperimentConfig;
    private valueNetwork!: tf.LayersModel;   // Сеть критика
    private policyNetwork!: tf.LayersModel;  // Сеть политики
    private policyOptimizer!: tf.Optimizer;        // Оптимизатор для policy network
    private valueOptimizer!: tf.Optimizer;         // Оптимизатор для value network

    private logger = new RingBuffer<{
        avgRewards: number;
        policyLoss: number;
        valueLoss: number;
    }>(1000);

    constructor() {
    }

    public static create() {
        return new MasterAgent().init();
    }

    getStats() {
        const last10Rewards = this.logger.getLastN(100).map(v => v.avgRewards);
        const avgReward = last10Rewards.length > 0
            ? last10Rewards.reduce((a, b) => a + b, 0) / last10Rewards.length
            : 0;

        const last10PolicyLoss = this.logger.getLastN(100).map(v => v.policyLoss);
        const avgPolicyLoss = last10PolicyLoss.length > 0
            ? last10PolicyLoss.reduce((a, b) => a + b, 0) / last10PolicyLoss.length
            : 0;

        const last10ValueLoss = this.logger.getLastN(100).map(v => v.valueLoss);
        const avgValueLoss = last10ValueLoss.length > 0
            ? last10ValueLoss.reduce((a, b) => a + b, 0) / last10ValueLoss.length
            : 0;

        return {
            avgReward: avgReward,
            avgPolicyLoss: avgPolicyLoss,
            avgValueLoss: avgValueLoss,
        };
    }

    // Сохранение модели
    async save() {
        try {
            await this.valueNetwork.save('indexeddb://tank-rl-value-model');
            await this.policyNetwork.save('indexeddb://tank-rl-policy-model');
            await setAgentState({
                config: this.config,
                logger: this.logger.toArray(),
            });
            return true;
        } catch (error) {
            console.error('Error saving models:', error);
            return false;
        }
    }

    async download() {
        return this.policyNetwork.save('downloads://tank-rl-policy-model');
    }

    async load() {
        try {
            const policyNetwork = await tf.loadLayersModel('indexeddb://tank-rl-policy-model');
            const valueNetwork = await tf.loadLayersModel('indexeddb://tank-rl-value-model');
            const agentState = await getAgentState();

            if (agentState && agentState.config) {
                console.log('Models loaded successfully');
                this.policyNetwork = policyNetwork;
                this.valueNetwork = valueNetwork;

                this.applyConfig(agentState.config);
                this.logger.fromArray(agentState.logger as any);

                return true;
            }

            return false;
        } catch (error) {
            console.warn('Could not load models, starting with new ones:', error);
            return false;
        }
    }

    predict(state: Float32Array): { action: Float32Array } {
        return tf.tidy(() => {
            const stateTensor = tf.tensor1d(state).expandDims(0);
            const predict = this.policyNetwork.predict(stateTensor) as tf.Tensor;
            const rawOutputSqueezed = predict.squeeze(); // [ACTION_DIM * 2] при batch=1
            const outMean = rawOutputSqueezed.slice([0], [ACTION_DIM]);   // ACTION_DIM штук

            return {
                action: outMean.dataSync() as Float32Array,
            };
        });
    }

    async tryTrain(): Promise<number> {
        const gradientsList = await getGradientsList();

        if (!gradientsList || gradientsList.length === 0) {
            return 0;
        }

        clearGradientsList();

        const prevWeights = isDevtoolsOpen() ? this.policyNetwork.getWeights().map(w => w.dataSync()) as Float32Array[] : null;

        for (const gradients of gradientsList) {
            this.applyGradientsPolicyNetwork(gradients.policy);
            this.applyGradientsValueNetwork(gradients.value);

            this.logger.add({
                avgRewards: gradients.avgReward,
                policyLoss: gradients.policy.loss,
                valueLoss: gradients.value.loss,
            });
        }

        const newWeights = isDevtoolsOpen() ? this.policyNetwork.getWeights().map(w => w.dataSync()) as Float32Array[] : null;

        isDevtoolsOpen() && console.log('>> WEIGHTS SUM ABS DELTA', newWeights!.reduce((acc, w, i) => {
            return acc + abs(w.reduce((a, b, j) => a + abs(b - prevWeights![i][j]), 0));
        }, 0));

        console.log(`[Train]: Applied ${ gradientsList.length } gradients`);

        return gradientsList.length;
    }

    private async init() {
        if (!(await this.load())) {
            this.applyConfig(getCurrentExperiment());
            this.policyNetwork = createPolicyNetwork(this.config.hiddenLayers);
            this.valueNetwork = createValueNetwork(this.config.hiddenLayers);
        }

        return this;
    }

    // Обучение сети политики
    private applyGradientsPolicyNetwork(
        gradsData: GradientData,
    ) {
        return tf.tidy(() => {
            const grads = Object.entries(gradsData.grads).reduce((acc, [name, tensorData]) => {
                // @ts-ignore
                acc[name] =
                    tf.tensor(tensorData.data, tensorData.shape);
                return acc;
            }, {});
            this.policyOptimizer.applyGradients(grads);
        });
    }

    // Обучение сети критика (оценка состояний)
    private applyGradientsValueNetwork(
        gradsData: GradientData,
    ) {
        return tf.tidy(() => {
            const grads = Object.entries(gradsData.grads).reduce((acc, [name, tensorData]) => {
                // @ts-ignore
                acc[name] =
                    tf.tensor(tensorData.data, tensorData.shape);
                return acc;
            }, {});
            this.valueOptimizer.applyGradients(grads);
        });
    }

    private applyConfig(config: RLExperimentConfig) {
        this.config = config;
        this.policyOptimizer = tf.train.adam(this.config.learningRatePolicy);
        this.valueOptimizer = tf.train.adam(this.config.learningRateValue);
    }
}

