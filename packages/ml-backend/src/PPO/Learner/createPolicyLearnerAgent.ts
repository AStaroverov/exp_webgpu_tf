import { getNetworkExpIteration, getNetworkLearningRate } from '../../../../ml-common/utils.ts';

import { createRequire } from 'module';
import * as tf from '../../../../ml-common/tf';

import { ceil, floor, max, mean, min } from '../../../../../lib/math.ts';
// import { metricsChannels } from '../../Common/channels.ts'; // metrics disabled
import { flatTypedArray } from '../../../../ml-common/flat.ts';
import { getDynamicLearningRate } from '../../../../ml-common/getDynamicLearningRate.ts';
import { createInputTensors } from '../../../../ml-common/InputTensors.ts';
import { ReplayBuffer } from '../../../../ml-common/ReplayBuffer.ts';
import { asyncUnwrapTensor, onReadyRead } from '../../../../ml-common/Tensor.ts';
import { createPolicyNetwork } from '../../Models/Create.ts';
import { Model } from '../../Models/def.ts';
import { CONFIG } from '../config.ts';
import { learningRateChannel } from '../localChannels.ts';
import { computeKullbackLeiblerExact, trainPolicyNetwork } from '../train.ts';
import { createLearnerAgent } from './createLearnerAgent.ts';
import { LearnData } from './createLearnerManager.ts';
import { isLossDangerous } from './isLossDangerous.ts';

const require = createRequire(import.meta.url);
const RingBufferModule = require('ring-buffer-ts');

export function createPolicyLearnerAgent() {
    return createLearnerAgent({
        modelName: Model.Policy,
        createNetwork: createPolicyNetwork,
        trainNetwork: trainPolicy,
    });
}

// @ts-ignore
const klHistory = new RingBufferModule.RingBuffer<number>(25);

function trainPolicy(network: tf.LayersModel, batch: LearnData) {
    const version = getNetworkExpIteration(network);
    const rb = new ReplayBuffer(batch.states.length);
    const mbs = CONFIG.miniBatchSize(version);
    const mbc = ceil(batch.size / mbs);

    console.info(`[Train Policy]: Stating..
         Iteration ${version},
         Sum batch size: ${batch.size},
         Mini batch count: ${mbc} by ${mbs}`);

    const getPolicyBatch = (batchSize: number, index: number) => {
        const indices = rb.getSample(batchSize, index * batchSize, (index + 1) * batchSize);
        return createPolicyBatch(batch, indices);
    };
    const getKLBatch = (size: number) => {
        const indices = rb.getSample(batch.size).slice(0, size);
        return createKlBatch(batch, indices);
    };

    const klSize = floor(mbs * ceil(mbc / 3));
    const klList: tf.Tensor[] = [];
    const policyLossList: tf.Tensor[] = [];
    const entropyCoeff = CONFIG.policyEntropy(version);

    for (let i = 0; i < CONFIG.policyEpochs(version); i++) {
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
                mbs,
                CONFIG.policyClipRatio,
                entropyCoeff,
                CONFIG.clipNorm,
                j === mbc - 1,
            );
            policyLoss && policyLossList.push(policyLoss);

            tf.dispose(tStates);
            tActions.dispose();
            tOldLogProbs.dispose();
            tAdvantages.dispose();
        }

        // KL
        const lkBatch = getKLBatch(klSize);
        const tStates = createInputTensors(lkBatch.states);
        const tMean = tf.tensor2d(flatTypedArray(lkBatch.mean), [lkBatch.mean.length, lkBatch.mean[0].length]);
        const tLogStd = tf.tensor2d(flatTypedArray(lkBatch.logStd), [lkBatch.logStd.length, lkBatch.logStd[0].length]);

        klList.push(computeKullbackLeiblerExact(
            network,
            tStates,
            tMean,
            tLogStd,
            mbs,
        ));

        tf.dispose(tStates);
        tMean.dispose();
        tLogStd.dispose();
    }

    return onReadyRead()
        .then(() => Promise.all([
            Promise.all(policyLossList.map((t) => asyncUnwrapTensor(t).then(v => v[0]))),
            Promise.all(klList.map((t) => asyncUnwrapTensor(t).then(v => v[0]))),
        ]))
        .then(([policyLossList, klList]) => {
            console.info(`[Train Policy]: Finish`);

            if (policyLossList.some((v) => isLossDangerous(v, 2))) {
                throw new Error(`Policy loss too dangerous: ${min(...policyLossList)}, ${max(...policyLossList)}`);
            }

            if (klList.some(kl => kl > CONFIG.klConfig.max)) {
                // throw new Error(`KL divergence too high ${max(...klList)}`);
            }

            klHistory.add(...klList);

            const lr = getDynamicLearningRate(
                mean(klHistory.toArray()),
                getNetworkLearningRate(network),
            );

            learningRateChannel.emit(lr);

            // metricsChannels.lr.postMessage([lr]);
            // metricsChannels.kl.postMessage(klList);
            // metricsChannels.policyLoss.postMessage(policyLossList);
        });
}

function createPolicyBatch(batch: LearnData, indices: number[]) {
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

function createKlBatch(batch: LearnData, indices: number[]) {
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
