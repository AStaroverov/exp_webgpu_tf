import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM, INPUT_DIM } from '../../../Common/consts.ts';
import { RingBuffer } from 'ring-buffer-ts';
import { isDevtoolsOpen } from '../../../Common/utils.ts';
import { abs } from '../../../../../../../lib/math.ts';
import { createPolicyNetwork, createValueNetwork } from '../../../Common/models.ts';
import { Config, getCurrentConfig } from '../../Common/config.ts';
import { getStoreModelPath } from '../utils.ts';
import {
    clearMemoryBatchList,
    getAgentState,
    getMemoryBatchCount,
    getMemoryBatchList,
    setAgentState,
} from '../Database.ts';
import { Batch } from '../../Common/Memory.ts';
import { trainPolicyNetwork, trainValueNetwork } from '../../Common/train.ts';
import { flatFloat32Array } from '../../../Common/flat.ts';

export class MasterAgent {
    private version = 0;
    private config!: Config;
    private valueNetwork!: tf.LayersModel;   // Сеть критика
    private policyNetwork!: tf.LayersModel;  // Сеть политики
    private policyOptimizer!: tf.Optimizer;        // Оптимизатор для policy network
    private valueOptimizer!: tf.Optimizer;         // Оптимизатор для value network

    private logger = new RingBuffer<{
        avgRewards: number;
        policyLoss: number;
        valueLoss: number;
    }>(10);

    constructor() {
        this.config = getCurrentConfig();
        this.valueOptimizer = tf.train.adam(this.config.learningRateValue);
        this.policyOptimizer = tf.train.adam(this.config.learningRatePolicy);
    }

    public static create() {
        return new MasterAgent().init();
    }

    getStats() {
        const last10Rewards = this.logger.getLastN(10).map(v => v.avgRewards);
        const avgReward = last10Rewards.length > 0
            ? last10Rewards.reduce((a, b) => a + b, 0) / last10Rewards.length
            : 0;

        const last10PolicyLoss = this.logger.getLastN(10).map(v => v.policyLoss);
        const avgPolicyLoss = last10PolicyLoss.length > 0
            ? last10PolicyLoss.reduce((a, b) => a + b, 0) / last10PolicyLoss.length
            : 0;

        const last10ValueLoss = this.logger.getLastN(10).map(v => v.valueLoss);
        const avgValueLoss = last10ValueLoss.length > 0
            ? last10ValueLoss.reduce((a, b) => a + b, 0) / last10ValueLoss.length
            : 0;

        return {
            version: this.version,
            avgReward10: avgReward,
            avgPolicyLoss10: avgPolicyLoss,
            avgValueLoss10: avgValueLoss,
            avgRewardLast: this.logger.getLast()?.avgRewards,
            avgPolicyLossLast: this.logger.getLast()?.policyLoss,
            avgValueLossLast: this.logger.getLast()?.valueLoss,
        };
    }

    // Сохранение модели
    async save() {
        try {
            this.version += 1;

            await setAgentState({
                version: this.version,
                config: this.config,
                logger: this.logger.toArray(),
            });
            await this.valueNetwork.save(getStoreModelPath('value-model', this.config));
            await this.policyNetwork.save(getStoreModelPath('policy-model', this.config));

            return true;
        } catch (error) {
            console.error('Error saving models:', error);
            return false;
        }
    }

    async download() {
        return Promise.all([
            this.valueNetwork.save(getStoreModelPath('value-model', this.config)),
            this.policyNetwork.save(getStoreModelPath('policy-model', this.config)),
        ]);
    }

    async load() {
        try {
            const agentState = await getAgentState();
            this.version = agentState?.version ?? 0;
            this.logger.fromArray(agentState?.logger as any);
            this.valueNetwork = await tf.loadLayersModel(getStoreModelPath('value-model', this.config));
            this.policyNetwork = await tf.loadLayersModel(getStoreModelPath('policy-model', this.config));
            console.log('Models loaded successfully');
            return true;
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
        const gradientsCount = await getMemoryBatchCount();

        if (gradientsCount < this.config.workerCount) {
            return 0;
        }

        const prevWeights = isDevtoolsOpen() ? this.policyNetwork.getWeights().map(w => w.dataSync()) as Float32Array[] : null;

        const batchList: Batch[] = await getMemoryBatchList();
        const size = batchList.reduce((acc, b) => acc + b.size, 0);
        const tStates = tf.tensor(flatFloat32Array(batchList.map(b => b.states)), [size, INPUT_DIM]);
        const tActions = tf.tensor(flatFloat32Array(batchList.map(b => b.actions)), [size, ACTION_DIM]);
        const tLogProbs = tf.tensor(batchList.map(b => b.logProbs).flat(), [size]);
        const tValues = tf.tensor(batchList.map(b => b.values).flat(), [size]);
        const tAdvantages = tf.tensor(batchList.map(b => b.advantages).flat(), [size]);
        const tReturns = tf.tensor(batchList.map(b => b.returns).flat(), [size]);
        let policyLossSum = 0, valueLossSum = 0;

        console.log(`[Train]: Iteration ${ this.version }, Batch size: ${ size }`);

        for (let i = 0; i < this.config.epochs; i++) {
            const policyLoss = trainPolicyNetwork(
                this.policyNetwork, this.policyOptimizer, this.config,
                tStates, tActions, tLogProbs, tAdvantages,
            );
            policyLossSum += policyLoss;

            const valueLoss = trainValueNetwork(
                this.valueNetwork, this.valueOptimizer, this.config,
                tStates, tReturns, tValues,
            );
            valueLossSum += valueLoss;

            console.log(`[Train]: Epoch: ${ i }, Policy loss: ${ policyLoss.toFixed(4) }, Value loss: ${ valueLoss.toFixed(4) }`);
        }

        clearMemoryBatchList();
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

        this.logger.add({
            avgRewards: batchList
                .map((b) => b.rewards.reduce((acc, v) => acc + v, 0) / b.rewards.length)
                .reduce((acc, v) => acc + v, 0) / batchList.length,
            policyLoss: policyLossSum / this.config.epochs,
            valueLoss: valueLossSum / this.config.epochs,
        });

        return gradientsCount;
    }

    private async init() {
        if (!(await this.load())) {
            this.policyNetwork = createPolicyNetwork(this.config.hiddenLayers);
            this.valueNetwork = createValueNetwork(this.config.hiddenLayers);
        }

        return this;
    }
}

