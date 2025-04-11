import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { floor, max, mean, min } from '../../../../../../lib/math.ts';
import { createPolicyNetwork } from '../../Models/Create.ts';
import { computeKullbackLeibler, trainPolicyNetwork } from '../train.ts';
import { CONFIG } from '../config.ts';
import { shuffle } from '../../../../../../lib/shuffle.ts';
import { RingBuffer } from 'ring-buffer-ts';
import { getDynamicLearningRate } from '../../Common/getDynamicLearningRate.ts';
import { forceExitChannel } from '../../Common/channels.ts';
import { sliceInputTensors } from '../../Common/InputTensors.ts';
import { BaseLearnerAgent } from './BaseLearnerAgent.ts';
import { Model } from '../../Models/Transfer.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';

export class PolicyLearnerAgent extends BaseLearnerAgent {
    private klHistory = new RingBuffer<number>(50);

    constructor() {
        super(createPolicyNetwork, Model.Policy);
    }

    public train(
        batchSize: number,
        miniBatchIndexes: number[],
        tAllStates: tf.Tensor[],
        tAllActions: tf.Tensor2D,
        tAllLogProbs: tf.Tensor1D,
        tAllAdvantages: tf.Tensor1D,
        onUpdateLR: (lr: number) => void,
    ) {
        const klList: number[] = [];
        const policyLossPromiseList: Promise<number>[] = [];

        for (let i = 0; i < CONFIG.policyEpochs; i++) {
            shuffle(miniBatchIndexes);
            for (let j = 0; j < miniBatchIndexes.length; j++) {
                const index = miniBatchIndexes[j];
                const start = index * CONFIG.miniBatchSize;
                const end = Math.min(start + CONFIG.miniBatchSize, batchSize);
                const size = end - start;
                const tStates = sliceInputTensors(tAllStates, start, size);
                const tActions = tAllActions.slice([start, 0], [size, -1]);
                const tLogProbs = tAllLogProbs.slice([start], [size]);
                const tAdvantages = tAllAdvantages.slice([start], [size]);

                policyLossPromiseList.push(trainPolicyNetwork(
                    this.network,
                    tStates, tActions, tLogProbs, tAdvantages,
                    CONFIG.clipRatio, CONFIG.entropyCoeff, CONFIG.clipNorm,
                ));

                tStates.forEach(t => t.dispose());
                tActions.dispose();
                tLogProbs.dispose();
                tAdvantages.dispose();
            }

            const klSize = max(min(CONFIG.miniBatchSize, batchSize), floor(batchSize / 3));
            const klStart = randomRangeInt(0, batchSize - klSize);
            const kl = computeKullbackLeibler(
                this.network,
                sliceInputTensors(tAllStates, klStart, klSize),
                tAllActions.slice([klStart, 0], [klSize, -1]),
                tAllLogProbs.slice([klStart], [klSize]),
            );

            if (kl > CONFIG.klConfig.max) {
                console.warn(`[Train]: KL divergence was too high: ${ kl }, in epoch ${ i }`);
                forceExitChannel.postMessage(null);
                break;
            }

            this.klHistory.add(kl);
            klList.push(kl);
        }

        const lr = getDynamicLearningRate(
            mean(this.klHistory.toArray()),
            getLR(this.network),
        );
        onUpdateLR(lr);

        return Promise.all(policyLossPromiseList).then((policyLossList) => {
            return {
                klList: klList,
                policyLossList: policyLossList,
            };
        });
    }
}

function getLR(o: tf.LayersModel): number {
    // @ts-expect-error
    return o.optimizer.learningRate;
}
