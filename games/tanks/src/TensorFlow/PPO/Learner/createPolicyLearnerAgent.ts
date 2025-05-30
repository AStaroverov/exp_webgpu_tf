import { getNetworkLearningRate, getNetworkVersion } from '../../Common/utils.ts';

import { createLearnerAgent } from './createLearnerAgent.ts';
import { createPolicyNetwork } from '../../Models/Create.ts';
import { CONFIG } from '../config.ts';
import * as tf from '@tensorflow/tfjs';
import { computeKullbackLeiblerExact, trainPolicyNetwork } from '../train.ts';
import { createInputTensors } from '../../Common/InputTensors.ts';
import { ReplayBuffer } from '../../Common/ReplayBuffer.ts';
import { ceil, floor, max, mean, min } from '../../../../../../lib/math.ts';
import { metricsChannels } from '../../Common/channels.ts';
import { flatTypedArray } from '../../Common/flat.ts';
import { getDynamicLearningRate } from '../../Common/getDynamicLearningRate.ts';
import { RingBuffer } from 'ring-buffer-ts';
import { learningRateChannel } from '../channels.ts';
import { LearnData } from './createLearnerManager.ts';
import { asyncUnwrapTensor, onReadyRead } from '../../Common/Tensor.ts';
import { isLossDangerous } from './isLossDangerous.ts';
import { Model } from '../../Models/def.ts';

export function createPolicyLearnerAgent() {
    return createLearnerAgent({
        modelName: Model.Policy,
        createNetwork: createPolicyNetwork,
        trainNetwork: trainPolicy,
    });
}

const klHistory = new RingBuffer<number>(25);

function trainPolicy(network: tf.LayersModel, batch: LearnData) {
    const version = getNetworkVersion(network);
    const rb = new ReplayBuffer(batch.states.length);
    const mbs = CONFIG.miniBatchSize;
    const mbc = ceil(batch.size / mbs);

    console.info(`[Train Policy]: Stating..
         Iteration ${ version },
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
    const klList: tf.Tensor[] = [];
    const policyLossList: tf.Tensor[] = [];
    const entropyCoeff = getEntropyCoeff(network.optimizer.iterations);

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
                throw new Error(`Policy loss too dangerous: ${ min(...policyLossList) }, ${ max(...policyLossList) }`);
            }

            if (klList.some(kl => kl > CONFIG.klConfig.max)) {
                throw new Error(`KL divergence too high ${ max(...klList) }`);
            }
            klHistory.add(...klList);

            const lr = getDynamicLearningRate(
                mean(klHistory.toArray()),
                getNetworkLearningRate(network),
            );

            learningRateChannel.emit(lr);

            metricsChannels.lr.postMessage([lr]);
            metricsChannels.kl.postMessage(klList);
            metricsChannels.policyLoss.postMessage(policyLossList);
        });
}

function getEntropyCoeff(iteration: number) {
    iteration = iteration % CONFIG.policyEntropy.reset;
    return CONFIG.policyEntropy.coeff * max(0, 1 - iteration / CONFIG.policyEntropy.limit);
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
