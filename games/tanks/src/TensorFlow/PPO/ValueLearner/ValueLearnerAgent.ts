import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ceil } from '../../../../../../lib/math.ts';
import { createValueNetwork } from '../../Common/models.ts';
import { getStoreModelPath } from '../../Common/tfUtils.ts';
import { policyAgentState, valueAgentState, valueMemory, ValueMemoryBatch } from '../../Common/Database.ts';
import { createInputTensors, sliceInputTensors, trainValueNetwork } from '../train.ts';
import { CONFIG } from '../config.ts';
import { batchShuffle, shuffle } from '../../../../../../lib/shuffle.ts';
import { setModelState } from '../../Common/modelsCopy.ts';
import { learningRateChannel, metricsChannels } from '../../Common/channels.ts';

export class ValueLearnerAgent {
    private version = 0;

    private batches: { version: number, memories: ValueMemoryBatch }[] = [];
    private lastTrainTimeStart = 0;

    private valueNetwork: tf.LayersModel = createValueNetwork();

    constructor() {
    }

    public async init() {
        if (!(await this.load())) {
            this.valueNetwork = createValueNetwork();
        }

        await this.startSyncLR();

        return this;
    }

    async save() {
        while (!(await this.upload())) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    async tryTrain(): Promise<boolean> {
        this.batches.push(...await valueMemory.extractMemoryBatchList());

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
        const values = new Float32Array(memories.map(b => b.values).flat());
        const returns = new Float32Array(memories.map(b => b.returns).flat());

        const miniBatchCount = ceil(states.length / CONFIG.miniBatchSize);
        console.log(`[Train]: Iteration ${ this.version }, Sum batch size: ${ sumSize }, Mini batch count: ${ miniBatchCount } by ${ CONFIG.miniBatchSize }`);

        batchShuffle(
            states,
            values,
            returns,
        );

        const tAllStates = createInputTensors(states);
        const tAllValues = tf.tensor1d(values);
        const tAllReturns = tf.tensor1d(returns);

        const valueLossPromises: Promise<number>[] = [];
        const miniBatchIndexes = Array.from({ length: miniBatchCount }, (_, i) => i);

        for (let i = 0; i < CONFIG.epochs; i++) {
            shuffle(miniBatchIndexes);
            for (let j = 0; j < miniBatchIndexes.length; j++) {
                const index = miniBatchIndexes[j];
                const start = index * CONFIG.miniBatchSize;
                const end = Math.min(start + CONFIG.miniBatchSize, states.length);
                const size = end - start;
                const tStates = sliceInputTensors(tAllStates, start, size);
                const tValues = tAllValues.slice([start], [size]);
                const tReturns = tAllReturns.slice([start], [size]);

                valueLossPromises.push(trainValueNetwork(
                    this.valueNetwork, this.valueNetwork.optimizer,
                    tStates, tReturns, tValues,
                    CONFIG.clipRatio,
                ));

                tStates.forEach(t => t.dispose());
                tValues.dispose();
                tReturns.dispose();
            }
        }

        const endTime = Date.now();
        Promise.all(valueLossPromises).then((valueLossList) => {
            for (let i = 0; i < CONFIG.epochs; i++) {
                let valueLoss = 0;
                for (let j = 0; j < miniBatchCount; j++) {
                    valueLoss += valueLossList[i * miniBatchCount + j];
                }

                valueLoss /= miniBatchCount;

                console.log('[Train]: Epoch', i, 'Value loss:', valueLoss);

                metricsChannels.valueLoss.postMessage(valueLoss);
            }

            metricsChannels.trainTime.postMessage((endTime - startTime) / 1000);
            metricsChannels.waitTime.postMessage(waitTime / 1000);
        });

        tAllStates.forEach(t => t.dispose());
        tAllValues.dispose();
        tAllReturns.dispose();

        this.version += 1;

        return true;
    }

    private async upload() {
        try {
            await Promise.all([
                valueAgentState.set({ version: this.version }),
                this.valueNetwork.save(getStoreModelPath('value-model', CONFIG), { includeOptimizer: true }),
            ]);

            return true;
        } catch (error) {
            console.error('Error saving models:', error);
            return false;
        }
    }

    private async load() {
        try {
            let [agentState, valueNetwork] = await Promise.all([
                valueAgentState.get(),
                tf.loadLayersModel(getStoreModelPath('value-model', CONFIG)),
            ]);

            // after change to new 2 threads we won't have valueAgentState
            if (valueNetwork != null && agentState == null) {
                agentState = await policyAgentState.get();
            }

            if (!valueNetwork) {
                return false;
            }
            this.version = agentState?.version ?? 0;
            this.valueNetwork = await setModelState(this.valueNetwork, valueNetwork);
            console.log('Models loaded successfully');

            valueNetwork.dispose();

            return true;
        } catch (error) {
            console.warn('Could not load models, starting with new ones:', error);
            return false;
        }
    }

    private async startSyncLR() {
        const policyState = await policyAgentState.get();
        if (policyState) {
            this.updateOptimizersLR(policyState.learningRate);
        }
        learningRateChannel.onmessage = (event) => {
            const lr = Number(event.data);
            isFinite(lr) && this.updateOptimizersLR(lr);
        };
    }

    private updateOptimizersLR(lr: number) {
        setLR(this.valueNetwork, lr);
    }
}

function setLR(o: tf.LayersModel, lr: number) {
    // @ts-expect-error
    o.optimizer.learningRate = lr;
}
