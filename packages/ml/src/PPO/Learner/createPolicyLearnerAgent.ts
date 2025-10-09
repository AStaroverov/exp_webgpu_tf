import { getNetworkExpIteration, getNetworkLearningRate, getNetworkPerturbConfig } from '../../../../ml-common/utils.ts';

import * as tf from '@tensorflow/tfjs';
import { RingBuffer } from 'ring-buffer-ts';
import { ceil, floor, mean, median, min } from '../../../../../lib/math.ts';
import { metricsChannels } from '../../../../ml-common/channels.ts';
import { CONFIG } from '../../../../ml-common/config.ts';
import { flatTypedArray } from '../../../../ml-common/flat.ts';
import { getDynamicLearningRate, getDynamicPerturb } from '../../../../ml-common/getDynamicLearningRate.ts';
import { createInputTensors } from '../../../../ml-common/InputTensors.ts';
import { ReplayBuffer } from '../../../../ml-common/ReplayBuffer.ts';
import { asyncUnwrapTensor, onReadyRead, syncUnwrapTensor } from '../../../../ml-common/Tensor.ts';
import { createPolicyNetwork } from '../../Models/Create.ts';
import { Model } from '../../Models/def.ts';
import { modelSettingsChannel } from '../channels.ts';
import { computeKullbackLeiblerExact, trainPolicyNetwork } from '../train.ts';
import { createLearnerAgent } from './createLearnerAgent.ts';
import { LearnData } from './createLearnerManager.ts';

export function createPolicyLearnerAgent() {
    return createLearnerAgent({
        modelName: Model.Policy,
        createNetwork: createPolicyNetwork,
        trainNetwork: trainPolicy,
    });
}

const klHistory = new RingBuffer<number>(25);
const klPerturbedHistory = new RingBuffer<number>(25);

function trainPolicy(network: tf.LayersModel, batch: LearnData) {
    const expIteration = getNetworkExpIteration(network);
    const mbs = CONFIG.miniBatchSize(expIteration);
    const mbc = ceil(batch.size / mbs);

    console.info(`[Train Policy]: Stating..
         Iteration ${expIteration},
         Sum batch size: ${batch.size},
         Mini batch count: ${mbc} by ${mbs}`);

    const getPolicyBatch = (() => {
        const rb = new ReplayBuffer(batch.states.length);

        return (batchSize: number, index: number) => {
            const indices = rb.getSample(batchSize, index * batchSize, (index + 1) * batchSize);
            return createPolicyBatch(batch, indices);
        };
    })()

    const createKLBatchGetter = (() => {
        const pureIndices: number[] = [];
        const perturbedIndices: number[] = [];
        for (let i = 0; i < batch.size; i++) {
            if (batch.perturbed[i] === 0) {
                pureIndices.push(i);
            } else {
                perturbedIndices.push(i);
            }
        }
        const pureRb = new ReplayBuffer(pureIndices.length);
        const perturbedRb = new ReplayBuffer(perturbedIndices.length);

        return (perturbed: boolean) => {
            return (size: number) => {
                const filteredIndices = perturbed ? perturbedIndices : pureIndices;
                if (filteredIndices.length === 0) return createKlBatch(batch, []);
                const rb = perturbed ? perturbedRb : pureRb;
                const indices = rb.getSample(min(size, filteredIndices.length));
                const mappedIndices = indices.map(i => filteredIndices[i]);
                return createKlBatch(batch, mappedIndices);
            };
        }
    })();

    // KL на данных без пертурбаций (для адаптации learning rate)
    const getKLBatch = createKLBatchGetter(false);
    // KL на данных с пертурбациями (только для метрик)
    const getKLPerturbedBatch = createKLBatchGetter(true);

    const klSize = floor(mbs * ceil(mbc / 3));

    const klList: number[] = [];
    const klPerturbedList: tf.Tensor[] = [];
    // const deviationList: tf.Tensor[] = [];
    // const actionDiffList: tf.Tensor[] = [];

    const policyLossList: tf.Tensor[] = [];
    const entropyCoeff = CONFIG.policyEntropy(expIteration);

    for (let i = 0; i < CONFIG.policyEpochs(expIteration); i++) {
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
                entropyCoeff,
                CONFIG.clipNorm,
                CONFIG.clipRhoPG,
                j === mbc - 1,
            );
            policyLoss && policyLossList.push(policyLoss);

            tf.dispose(tStates);
            tActions.dispose();
            tOldLogProbs.dispose();
            tAdvantages.dispose();
        }

        const tKL = computeKLForBatch(network, getKLBatch(klSize), mbs);
        const kl = tKL ? syncUnwrapTensor(tKL)[0] : undefined;
        if (kl) klList.push(kl);
        const tPerturbedKL = computeKLForBatch(network, getKLPerturbedBatch(klSize), mbs);
        if (tPerturbedKL) klPerturbedList.push(tPerturbedKL);

        if (kl && kl > CONFIG.klConfig.stopPure) {
            console.info(`Stopping policy training early at epoch ${i} due to high kl=${kl}`);
            break;
        }
    }

    return onReadyRead()
        .then(() => Promise.all([
            klList,
            Promise.all(klPerturbedList.map((t) => asyncUnwrapTensor(t).then(v => v[0]))),
            Promise.all(policyLossList.map((t) => asyncUnwrapTensor(t).then(v => v[0]))),
        ]))
        .then(([klList, klPerturbedList, policyLossList]) => {
            // if (policyLossList.some((v) => isLossDangerous(v, 2))) {
            //     throw new Error(`Policy loss too dangerous: ${min(...policyLossList)}, ${max(...policyLossList)}`);
            // }

            // if (klList.some(kl => kl > CONFIG.klConfig.maxPure)) {
            //     throw new Error(`KL divergence too high ${max(...klList)}`);
            // }

            // if (klPerturbedList.some(kl => kl > CONFIG.klConfig.maxPerturbed)) {
            //     throw new Error(`KL divergence on perturbed data too high ${max(...klPerturbedList)}`);
            // }

            klHistory.add(...klList);
            const klArr = klHistory.toArray();
            const kl = klArr.length > 0 ? (mean(klArr) + median(klArr)) / 2 : undefined;
            const lr = kl
                ? getDynamicLearningRate(kl, getNetworkLearningRate(network))
                : getNetworkLearningRate(network);

            klPerturbedHistory.add(...klPerturbedList);
            const klPerturbedArr = klPerturbedHistory.toArray();
            const klPerturbed = klPerturbedArr.length > 0 ? (mean(klPerturbedArr) + median(klPerturbedArr)) / 2 : undefined;
            const perturbScale = klPerturbed
                ? getDynamicPerturb(klPerturbed, getNetworkPerturbConfig(network).scale)
                : getNetworkPerturbConfig(network).scale;
            const perturbChance = 0;// kl === undefined || kl > CONFIG.lrConfig.kl.target ? 0 : CONFIG.perturbChance(expIteration);

            modelSettingsChannel.emit({ lr, perturbChance, perturbScale, expIteration: expIteration + batch.size });
            console.info(`[Train Policy]: Finish iteration=${expIteration}
                kl=${kl?.toFixed(3)}, lr=${lr},
                perturb kl=${klPerturbed?.toFixed(3)}, chance=${perturbChance.toFixed(2)}, scale=${perturbScale.toFixed(2)}`);

            metricsChannels.lr.postMessage([lr]);
            metricsChannels.kl.postMessage(klList);
            metricsChannels.klPerturbed.postMessage(klPerturbedList);
            metricsChannels.policyLoss.postMessage(policyLossList);
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
    const mean = indices.map(i => batch.mean[i]);
    const logStd = indices.map(i => batch.logStd[i]);

    return { states, mean, logStd };
}

function computeKLForBatch(
    network: tf.LayersModel,
    batch: ReturnType<typeof createKlBatch>,
    mbs: number,
) {
    let result: undefined | tf.Tensor;

    if (batch.states.length > 0) {
        const tStates = createInputTensors(batch.states);
        const tMean = tf.tensor2d(flatTypedArray(batch.mean), [batch.mean.length, batch.mean[0].length]);
        const tLogStd = tf.tensor2d(flatTypedArray(batch.logStd), [batch.logStd.length, batch.logStd[0].length]);

        result = computeKullbackLeiblerExact(
            network,
            tStates,
            tMean,
            tLogStd,
            mbs,
        )

        tf.dispose(tStates);
        tMean.dispose();
        tLogStd.dispose();
    }

    return result;
};