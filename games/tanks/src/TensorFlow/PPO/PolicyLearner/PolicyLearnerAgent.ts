import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM } from '../../Common/consts.ts';
import { ceil, mean } from '../../../../../../lib/math.ts';
import { createPolicyNetwork } from '../../Common/models.ts';
import { getStoreModelPath } from '../../Common/tfUtils.ts';
import { computeKullbackLeibler, healthCheck, trainPolicyNetwork } from '../train.ts';
import { CONFIG } from '../config.ts';
import { flatFloat32Array } from '../../Common/flat.ts';
import { batchShuffle, shuffle } from '../../../../../../lib/shuffle.ts';
import { RingBuffer } from 'ring-buffer-ts';
import { getDynamicLearningRate } from '../../Common/getDynamicLearningRate.ts';
import { setModelState } from '../../Common/modelsCopy.ts';
import { policyAgentState, policyMemory, PolicyMemoryBatch } from '../../Common/Database.ts';
import { learningRateChannel, metricsChannels } from '../../Common/channels.ts';
import { createInputTensors, sliceInputTensors } from '../../Common/InputTensors.ts';

export class PolicyLearnerAgent {
    private version = 0;
    private klHistory = new RingBuffer<number>(30);

    private batches: { version: number, memories: PolicyMemoryBatch }[] = [];
    private lastTrainTimeStart = 0;

    private policyNetwork: tf.LayersModel = createPolicyNetwork();

    constructor() {

    }

    public async init() {
        if (!(await this.load())) {
            this.policyNetwork = createPolicyNetwork();
        }

        return this;
    }

    async save() {
        while (!(await this.upload())) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    async tryTrain(): Promise<boolean> {
        this.batches.push(...await policyMemory.extractMemoryBatchList());

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

        batchShuffle(
            states,
            actions,
            logProbs,
            advantages,
        );

        const tAllStates = createInputTensors(states);
        const tAllActions = tf.tensor2d(flatFloat32Array(actions), [actions.length, ACTION_DIM]);
        const tAllLogProbs = tf.tensor1d(logProbs);
        const tAllAdvantages = tf.tensor1d(advantages);

        console.log(`[Train]: Iteration ${ this.version }, Sum batch size: ${ sumSize }, Mini batch count: ${ miniBatchCount } by ${ CONFIG.miniBatchSize }`);

        const policyLossPromises: Promise<number>[] = [];
        const klList: number[] = [];
        const miniBatchIndexes = Array.from({ length: miniBatchCount }, (_, i) => i);

        for (let i = 0; i < CONFIG.epochs; i++) {
            shuffle(miniBatchIndexes);
            for (let j = 0; j < miniBatchIndexes.length; j++) {
                const index = miniBatchIndexes[j];
                const start = index * CONFIG.miniBatchSize;
                const end = Math.min(start + CONFIG.miniBatchSize, states.length);
                const size = end - start;
                const tStates = sliceInputTensors(tAllStates, start, size);
                const tActions = tAllActions.slice([start, 0], [size, -1]);
                const tLogProbs = tAllLogProbs.slice([start], [size]);
                const tAdvantages = tAllAdvantages.slice([start], [size]);

                policyLossPromises.push(trainPolicyNetwork(
                    this.policyNetwork,
                    tStates, tActions, tLogProbs, tAdvantages,
                    CONFIG.clipRatio, CONFIG.entropyCoeff,
                ));

                tStates.forEach(t => t.dispose());
                tActions.dispose();
                tLogProbs.dispose();
                tAdvantages.dispose();
            }

            const kl = await computeKullbackLeibler(
                this.policyNetwork,
                tAllStates,
                tAllActions,
                tAllLogProbs,
            );

            if (kl > CONFIG.klConfig.max) {
                console.warn(`[Train]: KL divergence was too high: ${ kl }, in epoch ${ i }`);
                window.location.reload();
                break;
            }

            klList.push(kl);
            this.klHistory.add(kl);

            const lr = getDynamicLearningRate(
                mean(this.klHistory.toArray()),
                getLR(this.policyNetwork),
            );
            this.updateOptimizersLR(lr);
            metricsChannels.lr.postMessage(getLR(this.policyNetwork));
        }

        const endTime = Date.now();
        const version = this.version;
        Promise.all(policyLossPromises).then((policyLossList) => {
            for (let i = 0; i < klList.length; i++) {
                const kl = klList[i];

                let policyLoss = 0;
                for (let j = 0; j < miniBatchCount; j++) {
                    policyLoss += policyLossList[i * miniBatchCount + j];
                }

                policyLoss /= miniBatchCount;

                console.log('[Train]: Epoch', i, 'KL:', kl, 'Policy loss:', policyLoss);

                metricsChannels.kl.postMessage(kl);
                metricsChannels.policyLoss.postMessage(kl);
            }

            for (const batch of rawBatches) {
                metricsChannels.versionDelta.postMessage(version - batch.version);
                metricsChannels.batchSize.postMessage(batch.memories.size);
            }

            metricsChannels.advantages.postMessage(advantages);
            metricsChannels.trainTime.postMessage((endTime - startTime) / 1000);
            metricsChannels.waitTime.postMessage(waitTime / 1000);
            metricsChannels.rewards.postMessage(memories.map(b => b.rewards).flat());
        });

        tAllStates.forEach(t => t.dispose());
        tAllActions.dispose();
        tAllLogProbs.dispose();
        tAllAdvantages.dispose();

        this.version += 1;

        return true;
    }

    public healthCheck() {
        return healthCheck(this.policyNetwork);
    }

    public async load() {
        try {
            const [agentState, policyNetwork] = await Promise.all([
                policyAgentState.get(),
                tf.loadLayersModel(getStoreModelPath('policy-model', CONFIG)),
            ]);
            if (!policyNetwork) {
                return false;
            }
            this.version = agentState?.version ?? 0;
            this.klHistory.fromArray(agentState?.klHistory ?? []);
            this.policyNetwork = await setModelState(this.policyNetwork, policyNetwork);
            console.log('Models loaded successfully');

            policyNetwork.dispose();

            return true;
        } catch (error) {
            console.warn('Could not load models, starting with new ones:', error);
            return false;
        }
    }

    public async upload() {
        try {
            await Promise.all([
                policyAgentState.set({
                    version: this.version,
                    klHistory: this.klHistory.toArray(),
                    learningRate: getLR(this.policyNetwork),
                }),
                this.policyNetwork.save(getStoreModelPath('policy-model', CONFIG), { includeOptimizer: true }),
            ]);

            return true;
        } catch (error) {
            console.error('Error saving models:', error);
            return false;
        }
    }

    private updateOptimizersLR(lr: number) {
        setLR(this.policyNetwork, lr);
        learningRateChannel.postMessage(lr);
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
