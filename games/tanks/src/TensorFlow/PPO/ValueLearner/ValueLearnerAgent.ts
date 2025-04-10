import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { ceil } from '../../../../../../lib/math.ts';
import { createValueNetwork } from '../../Models/Create.ts';
import { trainValueNetwork } from '../train.ts';
import { CONFIG } from '../config.ts';
import { batchShuffle, shuffle } from '../../../../../../lib/shuffle.ts';
import { setModelState } from '../../Common/modelsCopy.ts';
import { learningRateChannel, metricsChannels, newValueVersionChannel } from '../../Common/channels.ts';
import { createInputTensors, sliceInputTensors } from '../../Common/InputTensors.ts';
import { LearnerAgent } from '../LearnerAgent.ts';
import { loadNetworkFromDB, Model, saveNetworkToDB } from '../../Models/Transfer.ts';

export class ValueLearnerAgent extends LearnerAgent {
    lastTrainTimeStart: number | null = null;

    constructor() {
        super(createValueNetwork, Model.Value);
    }

    public async init() {
        await super.init();
        await this.startSyncLR();
        return this;
    }

    public async train(batches = this.useBatches()) {
        const startTime = Date.now();
        const waitTime = startTime - (this.lastTrainTimeStart || startTime);
        this.lastTrainTimeStart = startTime;

        const version = this.getVersion();
        const memories = batches.map(b => b.memories);
        const sumSize = memories.reduce((acc, b) => acc + b.size, 0);
        const states = memories.map(b => b.states).flat();
        const values = new Float32Array(memories.map(b => b.values).flat());
        const returns = new Float32Array(memories.map(b => b.returns).flat());

        const miniBatchCount = ceil(states.length / CONFIG.miniBatchSize);
        console.log(`[Train]: Iteration ${ version }, Sum batch size: ${ sumSize }, Mini batch count: ${ miniBatchCount } by ${ CONFIG.miniBatchSize }`);

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
                    this.network,
                    tStates, tReturns, tValues,
                    CONFIG.clipRatio * 2, CONFIG.clipNorm,
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

            metricsChannels.values.postMessage(values);
            metricsChannels.returns.postMessage(returns);
            metricsChannels.trainTime.postMessage((endTime - startTime) / 1000);
            metricsChannels.waitTime.postMessage(waitTime / 1000);
        });

        tAllStates.forEach(t => t.dispose());
        tAllValues.dispose();
        tAllReturns.dispose();
    }

    public async upload() {
        try {
            await saveNetworkToDB(this.network, Model.Value);
            newValueVersionChannel.postMessage(this.getVersion());

            return true;
        } catch (error) {
            console.error('Error saving models:', error);
            return false;
        }
    }

    public async load() {
        try {
            let valueNetwork = await loadNetworkFromDB(Model.Value);

            if (!valueNetwork) return false;

            this.network = await setModelState(this.network, valueNetwork);

            valueNetwork.dispose();

            console.log('Models loaded successfully');
            return true;
        } catch (error) {
            console.warn('Could not load models, starting with new ones:', error);
            return false;
        }
    }

    private async startSyncLR() {
        learningRateChannel.addEventListener('message', (event) => {
            const lr = Number(event.data);
            isFinite(lr) && this.updateOptimizersLR(lr);
        });
    }
}
