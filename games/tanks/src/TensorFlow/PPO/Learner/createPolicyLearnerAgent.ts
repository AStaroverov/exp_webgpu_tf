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
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { flatTypedArray } from '../../Common/flat.ts';
import { getDynamicLearningRate } from '../../Common/getDynamicLearningRate.ts';
import { RingBuffer } from 'ring-buffer-ts';
import { learningRateChannel } from '../channels.ts';
import { LearnBatch } from './createLearnerManager.ts';

export function createPolicyLearnerAgent() {
    createLearnerAgent({
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
    const klList: number[] = [];
    const policyLossList: number[] = [];

    for (let i = 0; i < CONFIG.policyEpochs; i++) {
        for (let j = 0; j < mbc; j++) {
            const mBatch = getPolicyBatch(mbs, j);

            tf.tidy(() => {
                const policyLoss = trainPolicyNetwork(
                    network,
                    createInputTensors(mBatch.states),
                    tf.tensor2d(flatTypedArray(mBatch.actions), [mBatch.actions.length, mBatch.actions[0].length]),
                    tf.tensor1d(mBatch.logProbs),
                    tf.tensor1d(mBatch.advantages),
                    CONFIG.policyClipRatio, CONFIG.policyEntropyCoeff, CONFIG.clipNorm,
                    j === mbc - 1,
                );
                policyLoss && policyLossList.push(policyLoss);
            });
        }

        const lkBatch = getKLBatch(klSize);
        const kl = computeKullbackLeiblerExact(
            network,
            createInputTensors(lkBatch.states),
            tf.tensor2d(flatTypedArray(lkBatch.mean), [lkBatch.mean.length, lkBatch.mean[0].length]),
            tf.tensor2d(flatTypedArray(lkBatch.logStd), [lkBatch.logStd.length, lkBatch.logStd[0].length]),
        );

        if (kl > CONFIG.klConfig.max) {
            console.warn(`[Train]: KL divergence was too high: ${ kl }, in epoch ${ i }`);
            forceExitChannel.postMessage(null);
            break;
        }

        klHistory.add(kl);
        klList.push(kl);
    }

    const lr = getDynamicLearningRate(
        mean(klHistory.toArray()),
        getNetworkLearningRate(network),
    );

    learningRateChannel.emit(lr);
    metricsChannels.lr.postMessage(lr);

    macroTasks.addTimeout(() => {
        metricsChannels.kl.postMessage(klList);
        metricsChannels.policyLoss.postMessage(policyLossList);
    }, 0);
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
