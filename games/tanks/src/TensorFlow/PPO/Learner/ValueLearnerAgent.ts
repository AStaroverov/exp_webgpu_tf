import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { createValueNetwork } from '../../Models/Create.ts';
import { trainValueNetwork } from '../train.ts';
import { CONFIG } from '../config.ts';
import { createInputTensors } from '../../Common/InputTensors.ts';
import { BaseLearnerAgent } from './BaseLearnerAgent.ts';
import { Model } from '../../Models/Transfer.ts';
import { InputArrays } from '../../Common/InputArrays.ts';

export class ValueLearnerAgent extends BaseLearnerAgent {
    constructor() {
        super(createValueNetwork, Model.Value);
    }

    public train(
        batchCount: number,
        getBatch: (batchSize: number, index: number) => {
            states: InputArrays[],
            values: number[],
            returns: number[],
        },
    ): {
        valueLossList: number[],
    } {

        const valueLossList: number[] = [];

        for (let i = 0; i < CONFIG.valueEpochs; i++) {
            for (let j = 0; j < batchCount; j++) {
                const mBatch = getBatch(CONFIG.miniBatchSize, j);

                tf.tidy(() => {
                    const loss = trainValueNetwork(
                        this.network,
                        createInputTensors(mBatch.states),
                        tf.tensor1d(mBatch.returns),
                        tf.tensor1d(mBatch.values),
                        CONFIG.valueClipRatio, CONFIG.clipNorm,
                        j === batchCount - 1,
                    );

                    loss && valueLossList.push(loss);
                });
            }
        }

        return {
            valueLossList: valueLossList,
        };
    }
}
