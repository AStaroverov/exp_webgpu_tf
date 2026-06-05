import { getNetworkLearningRate, getNetworkSettings } from '../models/networkMeta.ts';

import * as tf from '@tensorflow/tfjs';
import { RingBuffer } from 'ring-buffer-ts';
import { ceil, floor, max, median, min } from '../../../../lib/math.ts';
import { metricsChannels } from '../infra/channels.ts';
import type { PpoConfig } from '../config.ts';
import { flatTypedArray } from '../utils/flat.ts';
import { getDynamicLearningRate } from '../utils/getDynamicLearningRate.ts';
import { ReplayBuffer } from '../memory/ReplayBuffer.ts';
import { asyncUnwrapTensor, onReadyRead, syncUnwrapTensor } from '../utils/Tensor.ts';
import { Model } from '../models/def.ts';
import { modelSettingsChannel } from '../core/channels.ts';
import { computeKullbackLeiblerAprox, trainPolicyNetwork } from '../core/train.ts';
import { createLearnerAgent } from './createLearnerAgent.ts';
import { LearnData } from './createLearnerManager.ts';
import { isLossDangerous } from './isLossDangerous.ts';

export function createPolicyLearnerAgent<S>({ config, createInputTensors, prepareRandomInputArrays, actionHeadDims, createNetwork, onNetworkReady }: {
    config: PpoConfig,
    createInputTensors: (batch: S[]) => tf.Tensor[],
    prepareRandomInputArrays: () => S,
    actionHeadDims: number[],
    createNetwork: () => tf.LayersModel,
    onNetworkReady?: (network: tf.LayersModel) => void,
}) {
    const totalDim = actionHeadDims.reduce((a, b) => a + b, 0);

    const klHistory = new RingBuffer<number>(25);

    const trainPolicy = (network: tf.LayersModel, batch: LearnData<S>) => {
        const settings = getNetworkSettings(network);
        const expIteration = settings.expIteration ?? 0;
        const mbs = config.miniBatchSize(expIteration);
        const mbc = ceil(batch.size / mbs);
        const rb = new ReplayBuffer(batch.states.length);

        console.info(`[Train Policy]: Stating..
             Iteration ${expIteration},
             Sum batch size: ${batch.size},
             Mini batch count: ${mbc} by ${mbs}`);

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

        for (let i = 0; i < config.policyEpochs(expIteration); i++) {
            for (let j = 0; j < mbc; j++) {
                const mBatch = getPolicyBatch(mbs, j);

                const tStates = createInputTensors(mBatch.states);
                const tActions = tf.tensor2d(flatTypedArray(mBatch.actions), [mBatch.actions.length, mBatch.actions[0].length]);
                const tOldLogProbs = tf.tensor1d(mBatch.logProbs);
                const tAdvantages = tf.tensor1d(mBatch.advantages);
                const tMasks = mBatch.masks != null
                    ? tf.tensor2d(flatTypedArray(mBatch.masks), [mBatch.masks.length, totalDim])
                    : undefined;

                const { loss, entropy } = trainPolicyNetwork(
                    network,
                    tStates,
                    tActions,
                    tOldLogProbs,
                    tAdvantages,
                    config.policyClipRatio,
                    config.entropyCoeff,
                    config.policyLogitsL2 ?? 0,
                    config.clipNorm,
                    j === mbc - 1,
                    tMasks,
                );
                loss && policyLossList.push(loss);
                entropyList.push(entropy);

                tf.dispose(tStates);
                tActions.dispose();
                tOldLogProbs.dispose();
                tAdvantages.dispose();
                tMasks?.dispose();
            }

            // KL on non-perturbed data (for learning rate adaptation)
            const tKL = computeKLForBatch(network, getKlBatch(klSize), mbs);
            const kl = tKL ? syncUnwrapTensor(tKL)[0] : undefined;
            if (kl != null) klList.push(kl);
            if (kl != null && kl > config.lrConfig.kl.high) {
                console.warn(`Stopping policy training early at epoch ${i} due to high KL=${kl}`);
                break;
            }
        }

        const avgEntropy = entropyList.reduce((a, b) => a + b, 0) / entropyList.length;

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
                    ? getDynamicLearningRate(kl, getNetworkLearningRate(network), config.lrConfig)
                    : getNetworkLearningRate(network);

                modelSettingsChannel.emit({ lr, expIteration: expIteration + batch.size });

                metricsChannels.kl.postMessage(klList);
                metricsChannels.lr.postMessage([lr]);
                metricsChannels.policyLoss.postMessage(policyLossList);
                metricsChannels.entropy.postMessage([avgEntropy]);

                console.info(`[Train Policy]: Finish iteration=${expIteration}, entropy=${avgEntropy.toFixed(4)}`);
            });
    };

    const createPolicyBatch = (batch: LearnData<S>, indices: number[]) => {
        const states = indices.map(i => batch.states[i]);
        const actions = indices.map(i => batch.actions[i]);
        const logProbs = indices.map(i => batch.logProbs[i]);
        const advantages = indices.map(i => batch.advantages[i]);
        const masks = batch.masks ? indices.map(i => batch.masks![i]) : undefined;

        return {
            states: states,
            actions: actions,
            logProbs: (logProbs),
            advantages: (advantages),
            masks: masks,
        };
    };

    const createKlBatch = (batch: LearnData<S>, indices: number[]) => {
        const states = indices.map(i => batch.states[i]);
        const actions = indices.map(i => batch.actions[i]);
        const logits = indices.map(i => batch.logits[i]);
        const logProb = indices.map(i => batch.logProbs[i]);
        const masks = batch.masks ? indices.map(i => batch.masks![i]) : undefined;

        return { states, actions, logits, logProb, masks };
    };

    const computeKLForBatch = (
        network: tf.LayersModel,
        batch: ReturnType<typeof createKlBatch>,
        mbs: number,
    ) => {
        let result: undefined | tf.Tensor;

        if (batch.states.length > 0) {
            const tStates = createInputTensors(batch.states);
            const tActions = tf.tensor2d(flatTypedArray(batch.actions), [batch.actions.length, batch.actions[0].length]);
            const tLogProb = tf.tensor1d(batch.logProb);
            const tMasks = batch.masks != null
                ? tf.tensor2d(flatTypedArray(batch.masks), [batch.masks.length, totalDim])
                : undefined;

            result = computeKullbackLeiblerAprox(
                network,
                tStates,
                tActions,
                tLogProb,
                mbs,
                tMasks,
            )

            tf.dispose(tStates);
            tActions.dispose();
            tLogProb.dispose();
            tMasks?.dispose();
        }

        return result;
    };

    return createLearnerAgent<S>({
        config,
        createInputTensors,
        prepareRandomInputArrays,
        modelName: Model.Policy,
        createNetwork,
        trainNetwork: trainPolicy,
        onNetworkReady,
    });
}
