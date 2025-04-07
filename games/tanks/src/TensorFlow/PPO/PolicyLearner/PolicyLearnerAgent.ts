import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ACTION_DIM } from '../../Common/consts.ts';
import { ceil, mean } from '../../../../../../lib/math.ts';
import { createPolicyNetwork } from '../../Models/Create.ts';
import { computeKullbackLeibler, trainPolicyNetwork } from '../train.ts';
import { CONFIG } from '../config.ts';
import { flatFloat32Array } from '../../Common/flat.ts';
import { batchShuffle, shuffle } from '../../../../../../lib/shuffle.ts';
import { RingBuffer } from 'ring-buffer-ts';
import { getDynamicLearningRate } from '../../Common/getDynamicLearningRate.ts';
import { setModelState } from '../../Common/modelsCopy.ts';
import { policyAgentState, policyMemory, PolicyMemoryBatch } from '../../Common/Database.ts';
import { learningRateChannel, metricsChannels, reloadChannel } from '../../Common/channels.ts';
import { createInputTensors, sliceInputTensors } from '../../Common/InputTensors.ts';
import { LearnerAgent } from '../LearnerAgent.ts';
import { loadNetwork, Model, saveNetwork } from '../../Models/Transfer.ts';

export class PolicyLearnerAgent extends LearnerAgent<{ version: number, memories: PolicyMemoryBatch }> {
    private klHistory = new RingBuffer<number>(30);
    private lastTrainTimeStart = 0;

    constructor() {
        super(createPolicyNetwork);
    }

    async collectBatches() {
        this.batches.push(...await policyMemory.extractMemoryBatchList());
    }

    async train(batches = this.extractBatches()) {
        const startTime = Date.now();
        const waitTime = startTime - (this.lastTrainTimeStart || startTime);
        this.lastTrainTimeStart = startTime;

        const memories = batches.map(b => b.memories);
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
                    this.network,
                    tStates, tActions, tLogProbs, tAdvantages,
                    CONFIG.clipRatio, CONFIG.entropyCoeff, CONFIG.clipNorm,
                ));

                tStates.forEach(t => t.dispose());
                tActions.dispose();
                tLogProbs.dispose();
                tAdvantages.dispose();
            }

            const kl = await computeKullbackLeibler(
                this.network,
                tAllStates,
                tAllActions,
                tAllLogProbs,
            );

            if (kl > CONFIG.klConfig.max) {
                console.warn(`[Train]: KL divergence was too high: ${ kl }, in epoch ${ i }`);
                reloadChannel.postMessage(null);
                break;
            }

            klList.push(kl);
            this.klHistory.add(kl);

            const lr = getDynamicLearningRate(
                mean(this.klHistory.toArray()),
                getLR(this.network),
            );
            this.updateOptimizersLR(lr);
            metricsChannels.lr.postMessage(getLR(this.network));
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

            for (const batch of batches) {
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
    }

    public async load() {
        try {
            const [agentState, network] = await Promise.all([
                policyAgentState.get(),
                loadNetwork(Model.Policy),
            ]);
            if (!network) {
                return false;
            }
            this.version = agentState?.version ?? 0;
            this.klHistory.fromArray(agentState?.klHistory ?? []);
            this.network = await setModelState(this.network, network);
            console.log('Models loaded successfully');

            network.dispose();

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
                    learningRate: getLR(this.network),
                }),
                saveNetwork(this.network, Model.Policy),
            ]);

            return true;
        } catch (error) {
            console.error('Error saving models:', error);
            return false;
        }
    }

    protected updateOptimizersLR(lr: number) {
        super.updateOptimizersLR(lr);
        learningRateChannel.postMessage(lr);
    }
}

function getLR(o: tf.LayersModel): number {
    // @ts-expect-error
    return o.optimizer.learningRate;
}