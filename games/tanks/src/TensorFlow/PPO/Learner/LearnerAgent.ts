import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM } from '../../Common/consts.ts';
import { ceil, mean } from '../../../../../../lib/math.ts';
import { createPolicyNetwork, createValueNetwork } from '../../Common/models.ts';
import { getStoreModelPath } from '../utils.ts';
import { extractMemoryBatchList, getAgentState, setAgentState } from '../Database.ts';
import { Batch } from '../Memory.ts';
import { computeKullbackLeibler, createInputTensors, trainPolicyNetwork, trainValueNetwork } from '../Common/train.ts';
import { CONFIG } from '../Common/config.ts';
import { flatFloat32Array } from '../../Common/flat.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { loadMetrics, logBatch, logEpoch, logLR, logRewards, logTrain, saveMetrics } from '../../Common/Metrics.ts';
import { batchShuffle } from '../../../../../../lib/shuffle.ts';
import { RingBuffer } from 'ring-buffer-ts';
import { getDynamicLearningRate } from '../../Common/getDynamicLearningRate.ts';
import { setModelState } from '../../Common/modelsCopy.ts';

export class LearnerAgent {
    private version = 0;
    private klHistory = new RingBuffer<number>(30);

    private batches: { version: number, memories: Batch }[] = [];
    private lastTrainTimeStart = 0;

    private policyNetwork: tf.LayersModel = createPolicyNetwork();
    private valueNetwork: tf.LayersModel = createValueNetwork();

    constructor() {

    }

    public static create() {
        return new LearnerAgent().init();
    }

    // Сохранение модели
    async save() {
        try {
            this.version += 1;

            await Promise.all([
                setAgentState({
                    version: this.version,
                    klHistory: this.klHistory.toArray(),
                }),
                this.valueNetwork.save(getStoreModelPath('value-model', CONFIG), { includeOptimizer: true }),
                this.policyNetwork.save(getStoreModelPath('policy-model', CONFIG), { includeOptimizer: true }),
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

    async tryTrain(): Promise<boolean> {
        this.batches.push(...await extractMemoryBatchList());

        if (this.batches.length < CONFIG.workerCount) {
            return false;
        }

        const startTime = Date.now();
        const waitTime = startTime - (this.lastTrainTimeStart || startTime);
        this.lastTrainTimeStart = startTime;

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

        const tAllStates = createInputTensors(states);
        const tAllActions = tf.tensor(flatFloat32Array(actions), [sumSize, ACTION_DIM]);
        const tAllLogProbs = tf.tensor(logProbs);

        console.log(`[Train]: Iteration ${ this.version }, Sum batch size: ${ sumSize }, Mini batch count: ${ miniBatchCount } by ${ CONFIG.miniBatchSize }`);

        const policyLossPromises: Promise<number>[] = [];
        const valueLossPromises: Promise<number>[] = [];
        const klList: number[] = [];

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
                const tStates = createInputTensors(states.slice(start, end));
                const tActions = tf.tensor(flatFloat32Array(actions.slice(start, end)), [size, ACTION_DIM]);
                const tLogProbs = tf.tensor(logProbs.subarray(start, end), [size]);
                const tValues = tf.tensor(values.subarray(start, end), [size]);
                const tAdvantages = tf.tensor(advantages.subarray(start, end), [size]);
                const tReturns = tf.tensor(returns.subarray(start, end), [size]);

                policyLossPromises.push(trainPolicyNetwork(
                    this.policyNetwork, this.policyNetwork.optimizer,
                    tStates, tActions, tLogProbs, tAdvantages,
                    CONFIG.clipRatio, CONFIG.entropyCoeff,
                ));

                valueLossPromises.push(trainValueNetwork(
                    this.valueNetwork, this.valueNetwork.optimizer,
                    tStates, tReturns, tValues,
                    CONFIG.clipRatio,
                ));

                macroTasks.addTimeout(() => {
                    tStates.forEach(t => t.dispose());
                    tActions.dispose();
                    tLogProbs.dispose();
                    tValues.dispose();
                    tAdvantages.dispose();
                    tReturns.dispose();
                }, 1000);
            }

            const kl = await computeKullbackLeibler(
                this.policyNetwork,
                tAllStates,
                tAllActions,
                tAllLogProbs,
            );

            klList.push(kl);

            // skip kl that too high
            if (kl < CONFIG.klConfig.target * 100) {
                this.klHistory.add(kl);
                const lr = getDynamicLearningRate(
                    mean(this.klHistory.toArray()),
                    getLR(this.policyNetwork),
                );
                this.updateOptimizersLR(lr);
            }

            logLR(getLR(this.policyNetwork));

            if (kl > CONFIG.klConfig.max) {
                console.warn(`[Train]: KL divergence was too high: ${ kl }, in epoch ${ i }`);
                break;
            }
        }

        const endTime = Date.now();
        const version = this.version;
        Promise.all([
            Promise.all(policyLossPromises),
            Promise.all(valueLossPromises),
        ]).then(([policyLossList, valueLossList]) => {
            for (let i = 0; i < klList.length; i++) {
                const kl = klList[i];

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
            logRewards(memories.map(b => b.rewards).flat());
            saveMetrics();
        });

        macroTasks.addTimeout(() => {
            tAllStates.forEach(t => t.dispose());
            tAllActions.dispose();
            tAllLogProbs.dispose();
        }, 1000);

        return true;
    }

    private async init() {
        if (!(await this.load())) {
            this.policyNetwork = createPolicyNetwork();
            this.valueNetwork = createValueNetwork();
        }

        // fake loss for save optimizer with model
        this.valueNetwork.loss = 'meanSquaredError';
        this.policyNetwork.loss = 'meanSquaredError';

        loadMetrics();

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
            this.klHistory.fromArray(agentState?.klHistory ?? []);
            this.valueNetwork = await setModelState(this.valueNetwork, valueNetwork);
            this.policyNetwork = await setModelState(this.policyNetwork, policyNetwork);
            console.log('[LearnAgent] Models loaded successfully');

            valueNetwork.dispose();
            policyNetwork.dispose();

            return true;
        } catch (error) {
            console.warn('[LearnAgent] Could not load models, starting with new ones:', error);
            return false;
        }
    }

    private updateOptimizersLR(lr: number) {
        setLR(this.policyNetwork, lr);
        setLR(this.valueNetwork, lr);
    }
}

function getLR(o: tf.LayersModel): number {
    // @ts-expect-error
    return o.optimizer.learningRate;
}

function setLR(o: tf.LayersModel, lr: number) {
    // @ts-expect-error
    o.optimizer.learningRate = lr;
}
