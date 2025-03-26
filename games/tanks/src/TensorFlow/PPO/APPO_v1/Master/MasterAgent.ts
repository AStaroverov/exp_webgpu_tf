import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM, INPUT_DIM } from '../../../Common/consts.ts';
import { RingBuffer } from 'ring-buffer-ts';
import { abs, ceil } from '../../../../../../../lib/math.ts';
import { createPolicyNetwork, createValueNetwork } from '../../../Common/models.ts';
import { getStoreModelPath } from '../utils.ts';
import { extractMemoryBatchList, getAgentLog, getAgentState, setAgentLog, setAgentState } from '../Database.ts';
import { Batch } from '../../Common/Memory.ts';
import { computeKullbackLeibler, predict, trainPolicyNetwork, trainValueNetwork } from '../../Common/train.ts';
import { CONFIG } from '../../Common/config.ts';
import { batchShuffle } from '../../../../../../../lib/shuffle.ts';
import { flatFloat32Array } from '../../../Common/flat.ts';
import { macroTasks } from '../../../../../../../lib/TasksScheduler/macroTasks.ts';

export class MasterAgent {
    private version = 0;
    private batches: Batch[] = [];
    private valueNetwork!: tf.LayersModel;   // Сеть критика
    private policyNetwork!: tf.LayersModel;  // Сеть политики
    private policyOptimizer!: tf.Optimizer;        // Оптимизатор для policy network
    private valueOptimizer!: tf.Optimizer;         // Оптимизатор для value network

    private logger = new RingBuffer<{
        avgBatchSize: number;
        avgRewards: number;
        policyLoss: number;
        valueLoss: number;
        avgKl: number;
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

        const last10Kl = this.logger.getLastN(10).map(v => v.avgKl);
        const avgKl = last10Kl.length > 0
            ? last10Kl.reduce((a, b) => a + b, 0) / last10Kl.length
            : 0;

        return {
            version: this.version,

            avgKL10: avgKl,
            avgKLLast: this.logger.getLast()?.avgKl,

            avgReward10: avgReward,
            avgRewardLast: this.logger.getLast()?.avgRewards,

            avgPolicyLoss10: avgPolicyLoss,
            avgPolicyLossLast: this.logger.getLast()?.policyLoss,

            avgValueLoss10: avgValueLoss,
            avgValueLossLast: this.logger.getLast()?.valueLoss,

            avgBatchSize10: avgBatchSize,
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
        return predict(
            this.policyNetwork,
            state,
        );
    }

    async tryTrain(): Promise<boolean> {
        this.batches.push(
            ...(await extractMemoryBatchList())
                .filter(b => {
                    const delta = this.version - b.version;
                    if (delta > 2) {
                        console.log('[Train]: skipping batch with diff', delta);
                        return false;
                    }
                    return true;
                })
                .map(b => b.memories),
        );

        if (this.batches.length < CONFIG.workerCount) {
            return false;
        }

        const batches = this.batches;
        this.batches = [];

        const sumSize = batches.reduce((acc, b) => acc + b.size, 0);
        const states = batches.map(b => b.states).flat();
        const actions = batches.map(b => b.actions).flat();
        const logProbs = new Float32Array(batches.map(b => b.logProbs).flat());
        const values = new Float32Array(batches.map(b => b.values).flat());
        const advantages = new Float32Array(batches.map(b => b.advantages).flat());
        const returns = new Float32Array(batches.map(b => b.returns).flat());
        const miniBatchCount = ceil(states.length / CONFIG.miniBatchSize);

        const tAllStates = tf.tensor(flatFloat32Array(states), [sumSize, INPUT_DIM]);
        const tAllActions = tf.tensor(flatFloat32Array(actions), [sumSize, ACTION_DIM]);
        const tAllLogProbs = tf.tensor(logProbs);

        console.log(`[Train]: Iteration ${ this.version }, Sum batch size: ${ sumSize }, Mini batch count: ${ miniBatchCount } by ${ CONFIG.miniBatchSize }`);

        const policyLossPromises: Promise<number>[] = [];
        const valueLossPromises: Promise<number>[] = [];
        const klPromises: Promise<number>[] = [];

        for (let i = 0; i < CONFIG.epochs; i++) {
            batchShuffle(
                states,
                actions,
                logProbs,
                values,
                advantages,
                returns,
            );

            for (let j = 0; j < miniBatchCount; j++) {
                const start = j * CONFIG.miniBatchSize;
                const end = Math.min(start + CONFIG.miniBatchSize, states.length);
                const size = end - start;
                const tStates = tf.tensor(flatFloat32Array(states).subarray(start * INPUT_DIM, end * INPUT_DIM), [size, INPUT_DIM]);
                const tActions = tf.tensor(flatFloat32Array(actions).subarray(start * ACTION_DIM, end * ACTION_DIM), [size, ACTION_DIM]);
                const tLogProbs = tf.tensor(logProbs.subarray(start, end), [size]);
                const tValues = tf.tensor(values.subarray(start, end), [size]);
                const tAdvantages = tf.tensor(advantages.subarray(start, end), [size]);
                const tReturns = tf.tensor(returns.subarray(start, end), [size]);

                policyLossPromises.push(trainPolicyNetwork(
                    this.policyNetwork, this.policyOptimizer, CONFIG,
                    tStates, tActions, tLogProbs, tAdvantages,
                ));

                valueLossPromises.push(trainValueNetwork(
                    this.valueNetwork, this.valueOptimizer, CONFIG,
                    tStates, tReturns, tValues,
                ));

                macroTasks.addTimeout(() => {
                    tStates.dispose();
                    tActions.dispose();
                    tLogProbs.dispose();
                    tValues.dispose();
                    tAdvantages.dispose();
                    tReturns.dispose();
                }, 1000);
            }

            klPromises.push(computeKullbackLeibler(
                this.policyNetwork,
                tAllStates,
                tAllActions,
                tAllLogProbs,
            ));
        }

        Promise.all([
            Promise.all(policyLossPromises),
            Promise.all(valueLossPromises),
            Promise.all(klPromises),
        ]).then(([policyLossList, valueLossList, klList]) => {
            let policyLossSum = 0, valueLossSum = 0, klSum = 0, count = 0;
            for (let i = 0; i < klList.length; i++) {
                const kl = klList[i];

                if (kl > CONFIG.maxKL) {
                    console.warn(`[Train]: KL divergence was too high: ${ kl }, in epoch ${ i }`);
                }

                let epochPolicyLossSum = 0, epochValueLossSum = 0;
                for (let j = 0; j < miniBatchCount; j++) {
                    epochPolicyLossSum += policyLossList[i * miniBatchCount + j];
                    epochValueLossSum += valueLossList[i * miniBatchCount + j];
                    count += 1;
                }

                console.log('[Train]: Epoch', i, 'KL:', kl, 'Policy loss:', epochPolicyLossSum, 'Value loss:', epochValueLossSum);

                policyLossSum += epochPolicyLossSum;
                valueLossSum += epochValueLossSum;
                klSum += abs(kl);
            }

            this.logger.add({
                avgBatchSize: sumSize / batches.length,
                avgRewards: this.batches
                    .map((b) => b.rewards.reduce((acc, v) => acc + v, 0) / b.rewards.length)
                    .reduce((acc, v) => acc + v, 0) / this.batches.length,
                policyLoss: policyLossSum / count,
                valueLoss: valueLossSum / count,
                avgKl: klSum / klList.length,
            });
        });

        macroTasks.addTimeout(() => {
            tAllStates.dispose();
            tAllActions.dispose();
            tAllLogProbs.dispose();
        }, 1000);

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
