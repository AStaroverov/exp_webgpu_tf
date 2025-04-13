import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { floor, mean } from '../../../../../../lib/math.ts';
import { createPolicyNetwork } from '../../Models/Create.ts';
import { computeKullbackLeibler, trainPolicyNetwork } from '../train.ts';
import { CONFIG } from '../config.ts';
import { RingBuffer } from 'ring-buffer-ts';
import { getDynamicLearningRate } from '../../Common/getDynamicLearningRate.ts';
import { forceExitChannel } from '../../Common/channels.ts';
import { createInputTensors } from '../../Common/InputTensors.ts';
import { BaseLearnerAgent } from './BaseLearnerAgent.ts';
import { Model } from '../../Models/Transfer.ts';
import { InputArrays } from '../../Common/InputArrays.ts';
import { flatTypedArray } from '../../Common/flat.ts';

export class PolicyLearnerAgent extends BaseLearnerAgent {
    private klHistory = new RingBuffer<number>(50);

    constructor() {
        super(createPolicyNetwork, Model.Policy);
    }

    public train(
        batchCount: number,
        getLearnBatch: (batchSize: number) => {
            states: InputArrays[],
            actions: Float32Array[],
            logProbs: number[],
            advantages: number[],
            // IS for prioritized replay
            weights: number[],
        },
        getKlBatch: (batchSize: number) => {
            states: InputArrays[],
            actions: Float32Array[],
            logProbs: number[],
        },
        onUpdateLR: (lr: number) => void,
    ): {
        klList: number[],
        policyLossList: number[],
    } {
        const klSize = floor(CONFIG.miniBatchSize / 3);
        const klList: number[] = [];
        const policyLossList: number[] = [];

        for (let i = 0; i < CONFIG.policyEpochs; i++) {
            for (let j = 0; j < batchCount; j++) {
                const mBatch = getLearnBatch(CONFIG.miniBatchSize);

                policyLossList.push(
                    tf.tidy(() => trainPolicyNetwork(
                        this.network,
                        createInputTensors(mBatch.states),
                        tf.tensor2d(flatTypedArray(mBatch.actions), [mBatch.actions.length, mBatch.actions[0].length]),
                        tf.tensor1d(mBatch.logProbs),
                        tf.tensor1d(mBatch.advantages),
                        tf.tensor1d(mBatch.weights),
                        CONFIG.clipRatio, CONFIG.entropyCoeff, CONFIG.clipNorm,
                    )));
            }

            const lkBatch = getKlBatch(klSize);
            const kl = computeKullbackLeibler(
                this.network,
                createInputTensors(lkBatch.states),
                tf.tensor2d(flatTypedArray(lkBatch.actions), [lkBatch.actions.length, lkBatch.actions[0].length]),
                tf.tensor1d(lkBatch.logProbs),
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

        return {
            klList: klList,
            policyLossList: policyLossList,
        };
    }
}

function getLR(o: tf.LayersModel): number {
    // @ts-expect-error
    return o.optimizer.learningRate;
}
