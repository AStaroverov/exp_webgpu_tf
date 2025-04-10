import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { createValueNetwork } from '../../Models/Create.ts';
import { trainValueNetwork } from '../train.ts';
import { CONFIG } from '../config.ts';
import { shuffle } from '../../../../../../lib/shuffle.ts';
import { sliceInputTensors } from '../../Common/InputTensors.ts';
import { BaseLearnerAgent } from './BaseLearnerAgent.ts';
import { Model } from '../../Models/Transfer.ts';

export class ValueLearnerAgent extends BaseLearnerAgent {
    constructor() {
        super(createValueNetwork, Model.Value);
    }

    public train(
        batchSize: number,
        miniBatchIndexes: number[],
        tAllStates: tf.Tensor[],
        tAllValues: tf.Tensor1D,
        tAllReturns: tf.Tensor1D,
    ) {
        const valueLossPromises: Promise<number>[] = [];

        for (let i = 0; i < CONFIG.epochs; i++) {
            shuffle(miniBatchIndexes);
            for (let j = 0; j < miniBatchIndexes.length; j++) {
                const index = miniBatchIndexes[j];
                const start = index * CONFIG.miniBatchSize;
                const end = Math.min(start + CONFIG.miniBatchSize, batchSize);
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

        return Promise.all(valueLossPromises).then((valueLossList) => {
            return {
                valueLossList,
            };
        });
    }
}
