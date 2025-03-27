import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM, INPUT_DIM } from '../../../Common/consts.ts';
import { ceil } from '../../../../../../../lib/math.ts';
import { createPolicyNetwork, createValueNetwork } from '../../../Common/models.ts';
import { getStoreModelPath } from '../utils.ts';
import { extractMemoryBatchList, getAgentState, setAgentState } from '../Database.ts';
import { Batch } from '../../Common/Memory.ts';
import { computeKullbackLeibler, predict, trainPolicyNetwork, trainValueNetwork } from '../../Common/train.ts';
import { CONFIG } from '../../Common/config.ts';
import { batchShuffle } from '../../../../../../../lib/shuffle.ts';
import { flatFloat32Array } from '../../../Common/flat.ts';
import { macroTasks } from '../../../../../../../lib/TasksScheduler/macroTasks.ts';
import { drawMetrics, logBatch, logEpoch, logRewards, logTrain, saveMetrics } from '../../Common/Metrics.ts';

export class MasterAgent {
    private version = 0;
    private batches: { version: number, memories: Batch }[] = [];
    private lastTrainTime = 0;

    private valueNetwork!: tf.LayersModel;   // Сеть критика
    private policyNetwork!: tf.LayersModel;  // Сеть политики
    private policyOptimizer!: tf.Optimizer;        // Оптимизатор для policy network
    private valueOptimizer!: tf.Optimizer;         // Оптимизатор для value network

    constructor() {
        this.valueOptimizer = tf.train.adam(CONFIG.learningRateValue);
        this.policyOptimizer = tf.train.adam(CONFIG.learningRatePolicy);
    }

    public static create() {
        return new MasterAgent().init();
    }

    getVersion() {
        return this.version;
    }

    // Сохранение модели
    async save() {
        try {
            this.version += 1;

            await Promise.all([
                setAgentState({ version: this.version, lastTrainTime: this.lastTrainTime }),
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
        this.batches.push(...await extractMemoryBatchList());

        if (this.batches.length < CONFIG.workerCount) {
            return false;
        }

        const startTime = Date.now();
        const waitTime = startTime - (this.lastTrainTime || startTime);
        this.lastTrainTime = startTime;

        const rawBatches = this.batches;
        const batches = rawBatches.filter(b => {
            const delta = this.version - b.version;
            if (delta > 2) {
                console.warn('[Train]: skipping batch with diff', delta);
                return false;
            }
            return true;
        });
        const memories = batches.map(b => b.memories);

        this.batches = [];
        const sumSize = memories.reduce((acc, b) => acc + b.size, 0);
        const states = memories.map(b => b.states).flat();
        const actions = memories.map(b => b.actions).flat();
        const logProbs = new Float32Array(memories.map(b => b.logProbs).flat());
        const values = new Float32Array(memories.map(b => b.values).flat());
        const returns = new Float32Array(memories.map(b => b.returns).flat());
        const advantages = new Float32Array(
            memories
                .map((b, i) => {
                    const lag = this.version - batches[i].version;
                    const trust = Math.max(0, 1 - CONFIG.trustCoeff * lag);
                    return b.advantages.map(a => a * trust);
                })
                .flat(),
        );

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

        const version = this.version;
        const endTime = Date.now();
        Promise.all([
            Promise.all(policyLossPromises),
            Promise.all(valueLossPromises),
            Promise.all(klPromises),
        ]).then(([policyLossList, valueLossList, klList]) => {
            for (let i = 0; i < klList.length; i++) {
                const kl = klList[i];

                if (kl > CONFIG.maxKL) {
                    console.warn(`[Train]: KL divergence was too high: ${ kl }, in epoch ${ i }`);
                }

                let policyLoss = 0, valueLoss = 0;
                for (let j = 0; j < miniBatchCount; j++) {
                    policyLoss += policyLossList[i * miniBatchCount + j];
                    valueLoss += valueLossList[i * miniBatchCount + j];
                }

                policyLoss /= miniBatchCount;
                valueLoss /= miniBatchCount;

                console.log('[Train]: Epoch', i, 'KL:', kl, 'Policy loss:', policyLoss, 'Value loss:', valueLoss);

                logEpoch({
                    kl,
                    valueLoss,
                    policyLoss,
                });
            }

            for (const batch of rawBatches) {
                logBatch({ versionDelta: version - batch.version, batchSize: batch.memories.size });
            }

            logTrain({ trainTime: (endTime - startTime) / 1000, waitTime: waitTime / 1000 });
            logRewards(memories.map(b => b.rewards.reduce((acc, v) => acc + v, 0) / b.rewards.length).flat());
            saveMetrics();
            drawMetrics();
        });

        macroTasks.addTimeout(() => {
            tAllStates.dispose();
            tAllActions.dispose();
            tAllLogProbs.dispose();
        }, 1000);

        this.lastTrainTime = Date.now();

        return true;
    }

    private async init() {
        if (!(await this.load())) {
            this.policyNetwork = createPolicyNetwork(CONFIG.hiddenLayersPolicy);
            this.valueNetwork = createValueNetwork(CONFIG.hiddenLayersValue);
        }

        return this;
    }

    private async load() {
        try {
            const [agentState, valueNetwork, policyNetwork] = await Promise.all([
                getAgentState(),
                tf.loadLayersModel(getStoreModelPath('value-model', CONFIG)),
                tf.loadLayersModel(getStoreModelPath('policy-model', CONFIG)),
            ]);

            if (!valueNetwork || !policyNetwork) {
                return false;
            }

            this.version = agentState?.version ?? 0;
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
