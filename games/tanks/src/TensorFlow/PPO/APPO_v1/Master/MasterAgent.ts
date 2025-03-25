import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM, INPUT_DIM } from '../../../Common/consts.ts';
import { RingBuffer } from 'ring-buffer-ts';
import { isDevtoolsOpen } from '../../../Common/utils.ts';
import { abs } from '../../../../../../../lib/math.ts';
import { createPolicyNetwork, createValueNetwork } from '../../../Common/models.ts';
import { getStoreModelPath } from '../utils.ts';
import { extractMemoryBatchList, getAgentLog, getAgentState, setAgentLog, setAgentState } from '../Database.ts';
import { Batch } from '../../Common/Memory.ts';
import { trainPolicyNetwork, trainValueNetwork } from '../../Common/train.ts';
import { flatFloat32Array } from '../../../Common/flat.ts';
import { CONFIG } from '../../Common/config.ts';

export class MasterAgent {
    private version = 0;
    private batches: Batch[] = [];
    private valueNetwork!: tf.LayersModel;   // Сеть критика
    private policyNetwork!: tf.LayersModel;  // Сеть политики
    private policyOptimizer!: tf.Optimizer;        // Оптимизатор для policy network
    private valueOptimizer!: tf.Optimizer;         // Оптимизатор для value network

    private logger = new RingBuffer<{
        avgRewards: number;
        policyLoss: number;
        valueLoss: number;
        avgBatchSize: number;
    }>(10);

    constructor() {
        this.valueOptimizer = tf.train.adam(CONFIG.learningRateValue);
        this.policyOptimizer = tf.train.adam(CONFIG.learningRatePolicy);
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

        const last10BatchSize = this.logger.getLastN(10).map(v => v.avgBatchSize);
        const avgBatchSize = last10BatchSize.length > 0
            ? last10BatchSize.reduce((a, b) => a + b, 0) / last10BatchSize.length
            : 0;

        return {
            version: this.version,
            avgReward10: avgReward,
            avgPolicyLoss10: avgPolicyLoss,
            avgValueLoss10: avgValueLoss,
            avgBatchSize10: avgBatchSize,
            avgRewardLast: this.logger.getLast()?.avgRewards,
            avgPolicyLossLast: this.logger.getLast()?.policyLoss,
            avgValueLossLast: this.logger.getLast()?.valueLoss,
            avgBatchSizeLast: this.logger.getLast()?.avgBatchSize,
        };
    }

    // Сохранение модели
    async save() {
        try {
            this.version += 1;

            await Promise.all([
                setAgentState({ version: this.version }),
                setAgentLog({ logger: this.logger.toArray() }),
                this.valueNetwork.save(getStoreModelPath('value-model', CONFIG)),
                this.policyNetwork.save(getStoreModelPath('policy-model', CONFIG)),
            ]);

            return true;
        } catch (error) {
            console.error('Error saving models:', error);
            return false;
        }
    }

    async download() {
        return Promise.all([
            this.valueNetwork.save(getStoreModelPath('value-model', CONFIG)),
            this.policyNetwork.save(getStoreModelPath('policy-model', CONFIG)),
        ]);
    }

    predict(state: Float32Array): { action: Float32Array } {
        return tf.tidy(() => {
            const stateTensor = tf.tensor1d(state).expandDims(0);
            const predict = this.policyNetwork.predict(stateTensor) as tf.Tensor;
            const rawOutputSqueezed = predict.squeeze(); // [ACTION_DIM * 2] при batch=1
            const outMean = rawOutputSqueezed.slice([0], [ACTION_DIM]);   // ACTION_DIM штук

            return {
                action: outMean.tanh().dataSync() as Float32Array,
            };
        });
    }

    async tryTrain(): Promise<boolean> {
        this.batches.push(...await extractMemoryBatchList());

        const size = this.batches.reduce((acc, b) => acc + b.size, 0);

        if (size < CONFIG.batchSize) {
            return false;
        }

        const tStates = tf.tensor(flatFloat32Array(this.batches.map(b => b.states)), [size, INPUT_DIM]);
        const tActions = tf.tensor(flatFloat32Array(this.batches.map(b => b.actions)), [size, ACTION_DIM]);
        const tLogProbs = tf.tensor(this.batches.map(b => b.logProbs).flat(), [size]);
        const tValues = tf.tensor(this.batches.map(b => b.values).flat(), [size]);
        const tAdvantages = tf.tensor(this.batches.map(b => b.advantages).flat(), [size]);
        const tReturns = tf.tensor(this.batches.map(b => b.returns).flat(), [size]);
        let policyLossSum = 0, valueLossSum = 0;

        console.log(`[Train]: Iteration ${ this.version }, Batch size: ${ size }`);
        const prevWeights = isDevtoolsOpen() ? this.policyNetwork.getWeights().map(w => w.dataSync()) as Float32Array[] : null;

        for (let i = 0; i < CONFIG.epochs; i++) {
            const policyLoss = trainPolicyNetwork(
                this.policyNetwork, this.policyOptimizer, CONFIG,
                tStates, tActions, tLogProbs, tAdvantages,
            );
            policyLossSum += policyLoss;

            const valueLoss = trainValueNetwork(
                this.valueNetwork, this.valueOptimizer, CONFIG,
                tStates, tReturns, tValues,
            );
            valueLossSum += valueLoss;

            console.log(`[Train]: Epoch: ${ i }, Policy loss: ${ policyLoss.toFixed(4) }, Value loss: ${ valueLoss.toFixed(4) }`);
        }

        const newWeights = isDevtoolsOpen() ? this.policyNetwork.getWeights().map(w => w.dataSync()) as Float32Array[] : null;

        isDevtoolsOpen() && console.log('>> WEIGHTS SUM ABS DELTA', newWeights!.reduce((acc, w, i) => {
            return acc + abs(w.reduce((a, b, j) => a + abs(b - prevWeights![i][j]), 0));
        }, 0));

        this.logger.add({
            avgRewards: this.batches
                .map((b) => b.rewards.reduce((acc, v) => acc + v, 0) / b.rewards.length)
                .reduce((acc, v) => acc + v, 0) / this.batches.length,
            policyLoss: policyLossSum / CONFIG.epochs,
            valueLoss: valueLossSum / CONFIG.epochs,
            avgBatchSize: size / this.batches.length,
        });

        this.batches.length = 0;
        tStates.dispose();
        tActions.dispose();
        tLogProbs.dispose();
        tValues.dispose();
        tAdvantages.dispose();
        tReturns.dispose();

        return true;
    }

    private async init() {
        if (!(await this.load())) {
            this.policyNetwork = createPolicyNetwork(CONFIG.hiddenLayers);
            this.valueNetwork = createValueNetwork(CONFIG.hiddenLayers);
        }

        return this;
    }

    private async load() {
        try {
            const [agentState, agentLog, valueNetwork, policyNetwork] = await Promise.all([
                getAgentState(),
                getAgentLog(),
                tf.loadLayersModel(getStoreModelPath('value-model', CONFIG)),
                tf.loadLayersModel(getStoreModelPath('policy-model', CONFIG)),
            ]);

            if (!valueNetwork || !policyNetwork) {
                return false;
            }

            this.version = agentState?.version ?? 0;
            this.logger.fromArray(agentLog?.logger ?? [] as any);
            this.valueNetwork = valueNetwork;
            this.policyNetwork = policyNetwork;
            console.log('[MasterAgent] Models loaded successfully');
            return true;
        } catch (error) {
            console.warn('[MasterAgent] Could not load models, starting with new ones:', error);
            return false;
        }
    }
}



