import { getNetworkLearningRate, getNetworkSettings } from '../../../../ml-common/utils.ts';

import * as tf from '@tensorflow/tfjs';
import { clamp } from 'lodash-es';
import { RingBuffer } from 'ring-buffer-ts';
import { ceil, floor, max, median, min } from '../../../../../lib/math.ts';
import { metricsChannels } from '../../../../ml-common/channels.ts';
import { CONFIG } from '../../../../ml-common/config.ts';
import { flatTypedArray } from '../../../../ml-common/flat.ts';
import { getDynamicLearningRate } from '../../../../ml-common/getDynamicLearningRate.ts';
import { createInputTensors } from '../../../../ml-common/InputTensors.ts';
import { ReplayBuffer } from '../../../../ml-common/ReplayBuffer.ts';
import { asyncUnwrapTensor, onReadyRead, syncUnwrapTensor } from '../../../../ml-common/Tensor.ts';
import { ACTION_HEAD_DIMS, createPolicyNetwork } from '../../Models/Create.ts';
import { Model } from '../../Models/def.ts';
import { modelSettingsChannel } from '../channels.ts';
import { computeKullbackLeiblerAprox, trainPolicyNetwork } from '../train.ts';
import { createLearnerAgent } from './createLearnerAgent.ts';
import { LearnData } from './createLearnerManager.ts';
import { isLossDangerous } from './isLossDangerous.ts';

// Compute target entropy from action head dimensions
// targetEntropy = targetRatio * mean(log(numClasses_i))
const maxEntropyPerHead = ACTION_HEAD_DIMS.map(dim => Math.log(dim));
const meanMaxEntropy = maxEntropyPerHead.reduce((a, b) => a + b, 0) / maxEntropyPerHead.length;
const targetEntropy = CONFIG.adaptiveEntropy.targetRatio * meanMaxEntropy;

export function createPolicyLearnerAgent() {
    return createLearnerAgent({
        modelName: Model.Policy,
        createNetwork: createPolicyNetwork,
        trainNetwork: trainPolicy,
    });
}

const klHistory = new RingBuffer<number>(25);

function trainPolicy(network: tf.LayersModel, batch: LearnData) {
    const settings = getNetworkSettings(network);
    const expIteration = settings.expIteration ?? 0;
    const mbs = CONFIG.miniBatchSize(expIteration);
    const mbc = ceil(batch.size / mbs);
    const rb = new ReplayBuffer(batch.states.length);

    // Adaptive entropy coefficient: adapts to keep H(π) near target
    let logEntropyCoeff = settings.logEntropyCoeff ?? CONFIG.adaptiveEntropy.initialLogAlpha;
    const entropyCoeff = Math.exp(logEntropyCoeff);

    console.info(`[Train Policy]: Stating..
         Iteration ${expIteration},
         Sum batch size: ${batch.size},
         Mini batch count: ${mbc} by ${mbs},
         Entropy α=${entropyCoeff.toFixed(4)}`);

    const getPolicyBatch = (batchSize: number, index: number) => {
        const indices = rb.getSample(batchSize, index * batchSize, (index + 1) * batchSize);
        return createPolicyBatch(batch, indices);
    };
    const getKlBatch = (batchSize: number) => {
        const indices = rb.getSample(batchSize);
        return createKlBatch(batch, indices);
    }

    const klSize = floor(mbs * ceil(mbc / 3));
    const klList: number[] = [];
    const entropyList: number[] = [];
    const policyLossList: tf.Tensor[] = [];

    for (let i = 0; i < CONFIG.policyEpochs(expIteration); i++) {
        for (let j = 0; j < mbc; j++) {
            const mBatch = getPolicyBatch(mbs, j);

            const tStates = createInputTensors(mBatch.states);
            const tActions = tf.tensor2d(flatTypedArray(mBatch.actions), [mBatch.actions.length, mBatch.actions[0].length]);
            const tOldLogProbs = tf.tensor1d(mBatch.logProbs);
            const tAdvantages = tf.tensor1d(mBatch.advantages);

            const { loss, entropy } = trainPolicyNetwork(
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
            loss && policyLossList.push(loss);
            entropyList.push(entropy);

            tf.dispose(tStates);
            tActions.dispose();
            tOldLogProbs.dispose();
            tAdvantages.dispose();
        }

        // KL on non-perturbed data (for learning rate adaptation)
        const tKL = computeKLForBatch(network, getKlBatch(klSize), mbs);
        const kl = tKL ? syncUnwrapTensor(tKL)[0] : undefined;
        if (kl != null) klList.push(kl);
        if (kl != null && kl > CONFIG.lrConfig.kl.high) {
            console.warn(`Stopping policy training early at epoch ${i} due to high KL=${kl}`);
            break;
        }
    }

    // Update log_entropyCoeff based on average entropy vs target
    const avgEntropy = entropyList.reduce((a, b) => a + b, 0) / entropyList.length;
    const newLogEntropyCoeff = clamp(
        logEntropyCoeff - CONFIG.adaptiveEntropy.alphaLR * (avgEntropy - targetEntropy),
        CONFIG.adaptiveEntropy.minLogAlpha,
        CONFIG.adaptiveEntropy.maxLogAlpha
    );

    return onReadyRead()
        .then(() => Promise.all([
            klList,
            Promise.all(policyLossList.map((t) => asyncUnwrapTensor(t).then(v => v[0]))),
        ]))
        .then(([klList, policyLossList]) => {
            if (policyLossList.some((v) => isLossDangerous(v, 1000))) {
                throw new Error(`Policy loss too dangerous: ${min(...policyLossList)}, ${max(...policyLossList)}`);
            }

            klHistory.add(...klList);

            const klHistoryList = klHistory.toArray();
            const kl = klHistoryList.length > 0 ? median(klHistoryList) : undefined;
            const lr = kl != null
                ? getDynamicLearningRate(kl, getNetworkLearningRate(network))
                : getNetworkLearningRate(network);

            modelSettingsChannel.emit({ lr, expIteration: expIteration + batch.size, logEntropyCoeff: newLogEntropyCoeff });

            metricsChannels.kl.postMessage(klList);
            metricsChannels.lr.postMessage([lr]);
            metricsChannels.policyLoss.postMessage(policyLossList);
            metricsChannels.entropy.postMessage([avgEntropy]);
            metricsChannels.entropyAlpha.postMessage([Math.exp(newLogEntropyCoeff)]);

            console.info(`[Train Policy]: Finish iteration=${expIteration}, entropy=${avgEntropy.toFixed(4)}, α=${Math.exp(newLogEntropyCoeff).toFixed(4)}`);
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
    const logits = indices.map(i => batch.logits[i]);
    const logProb = indices.map(i => batch.logProbs[i]);

    return { states, actions, logits, logProb, };
}

function computeKLForBatch(
    network: tf.LayersModel,
    batch: ReturnType<typeof createKlBatch>,
    mbs: number,
) {
    let result: undefined | tf.Tensor;

    if (batch.states.length > 0) {
        const tStates = createInputTensors(batch.states);
        const tActions = tf.tensor2d(flatTypedArray(batch.actions), [batch.actions.length, batch.actions[0].length]);
        const tLogProb = tf.tensor1d(batch.logProb);

        result = computeKullbackLeiblerAprox(
            network,
            tStates,
            tActions,
            tLogProb,
            mbs,
        )

        tf.dispose(tStates);
        tActions.dispose();
        tLogProb.dispose();
    }

    return result;
};