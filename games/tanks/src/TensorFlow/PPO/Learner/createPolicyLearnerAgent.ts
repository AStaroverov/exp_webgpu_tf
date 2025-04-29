import { getNetworkLearningRate, getNetworkVersion } from '../../Common/utils.ts';

import { Model } from '../../Models/Transfer.ts';
import { createLearnerAgent } from './createLearnerAgent.ts';
import { createPolicyNetwork } from '../../Models/Create.ts';
import { CONFIG } from '../config.ts';
import * as tf from '@tensorflow/tfjs';
import { computeKullbackLeiblerExact, trainPolicyNetwork } from '../train.ts';
import { createInputTensors } from '../../Common/InputTensors.ts';
import { ReplayBuffer } from '../../Common/ReplayBuffer.ts';
import { ceil, floor, mean } from '../../../../../../lib/math.ts';
import { forceExitChannel, metricsChannels } from '../../Common/channels.ts';
import { flatTypedArray } from '../../Common/flat.ts';
import { getDynamicLearningRate } from '../../Common/getDynamicLearningRate.ts';
import { RingBuffer } from 'ring-buffer-ts';
import { learningRateChannel } from '../channels.ts';
import { LearnBatch } from './createLearnerManager.ts';

export function createPolicyLearnerAgent() {
    return createLearnerAgent({
        modelName: Model.Policy,
        createNetwork: createPolicyNetwork,
        trainNetwork: trainPolicy,
    });
}

const klHistory = new RingBuffer<number>(25);

function trainPolicy(network: tf.LayersModel, batch: LearnBatch) {
    const version = getNetworkVersion(network);
    const rb = new ReplayBuffer(batch.states.length);
    const mbs = CONFIG.miniBatchSize;
    const mbc = ceil(batch.size / mbs);

    console.log(`[Train Policy]: Iteration ${ version },
         Sum batch size: ${ batch.size },
         Mini batch count: ${ mbc } by ${ mbs }`);

    const getPolicyBatch = (batchSize: number, index: number) => {
        const indices = rb.getSample(batchSize, index * batchSize, (index + 1) * batchSize);
        return createPolicyBatch(batch, indices);
    };
    const getKLBatch = (size: number) => {
        const indices = rb.getSample(batch.size).slice(0, size);
        return createKlBatch(batch, indices);
    };

    const klSize = floor(mbs * ceil(mbc / 3));
    const klList: Promise<number>[] = [];
    const policyLossList: Promise<number>[] = [];

    for (let i = 0; i < CONFIG.policyEpochs; i++) {
        for (let j = 0; j < mbc; j++) {
            const mBatch = getPolicyBatch(mbs, j);

            const tStates = createInputTensors(mBatch.states);
            const tActions = tf.tensor2d(flatTypedArray(mBatch.actions), [mBatch.actions.length, mBatch.actions[0].length]);
            const tOldLogProbs = tf.tensor1d(mBatch.logProbs);
            const tAdvantages = tf.tensor1d(mBatch.advantages);

            const policyLoss = trainPolicyNetwork(
                network,
                tStates,
                tActions,
                tOldLogProbs,
                tAdvantages,
                CONFIG.policyClipRatio, CONFIG.policyEntropyCoeff, CONFIG.clipNorm,
                j === mbc - 1,
            );
            policyLoss && policyLossList.push(policyLoss);

            tf.dispose(tStates);
            tActions.dispose();
            tOldLogProbs.dispose();
            tAdvantages.dispose();
        }

        const lkBatch = getKLBatch(klSize);
        const klPromise = computeKullbackLeiblerExact(
            network,
            createInputTensors(lkBatch.states),
            tf.tensor2d(flatTypedArray(lkBatch.mean), [lkBatch.mean.length, lkBatch.mean[0].length]),
            tf.tensor2d(flatTypedArray(lkBatch.logStd), [lkBatch.logStd.length, lkBatch.logStd[0].length]),
        );

        klList.push(klPromise);
        klPromise.then((kl) => {
            if (kl > CONFIG.klConfig.max) {
                console.warn(`[Train]: KL divergence was too high: ${ kl }, in epoch ${ i }`);
                forceExitChannel.postMessage(null);
            }
        });
    }

    Promise.all([
        Promise.all(klList),
        Promise.all(policyLossList),
    ]).then(([klList, policyLossList]) => {
        klHistory.add(...klList);

        const lr = getDynamicLearningRate(
            mean(klHistory.toArray()),
            getNetworkLearningRate(network),
        );

        learningRateChannel.emit(lr);

        metricsChannels.lr.postMessage(lr);
        metricsChannels.kl.postMessage(klList);
        metricsChannels.policyLoss.postMessage(policyLossList);
    });
}

function createPolicyBatch(batch: LearnBatch, indices: number[]) {
    const states = indices.map(i => batch.states[i]);
    const actions = indices.map(i => batch.actions[i]);
    const logProbs = indices.map(i => batch.logProbs[i]);
    const advantages = indices.map(i => batch.advantages[i]);

    return {
        states: states,
        actions: actions,
        logProbs: (logProbs),
        advantages: (advantages),
    };
}

function createKlBatch(batch: LearnBatch, indices: number[]) {
    const states = indices.map(i => batch.states[i]);
    const actions = indices.map(i => batch.actions[i]);
    const mean = indices.map(i => batch.mean[i]);
    const logStd = indices.map(i => batch.logStd[i]);

    return {
        states: states,
        actions: actions,
        mean: (mean),
        logStd: (logStd),
    };
}
