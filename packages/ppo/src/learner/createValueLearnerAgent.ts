import * as tf from '@tensorflow/tfjs';
import { ceil, max, min } from '../../../../lib/math.ts';
import { metricsChannels } from '../infra/channels.ts';
import type { PpoConfig } from '../config.ts';
import { ReplayBuffer } from '../memory/ReplayBuffer.ts';
import { asyncUnwrapTensor, onReadyRead } from '../utils/Tensor.ts';
import { getNetworkExpIteration } from '../models/networkMeta.ts';
import { Model } from '../models/def.ts';
import { trainValueNetwork } from '../core/train.ts';
import { createLearnerAgent } from './createLearnerAgent.ts';
import { LearnData } from './createLearnerManager.ts';
import { isLossDangerous } from './isLossDangerous.ts';

export function createValueLearnerAgent<S>({ config, createInputTensors, prepareRandomInputArrays, createNetwork, onNetworkReady }: {
    config: PpoConfig,
    createInputTensors: (batch: S[]) => tf.Tensor[],
    prepareRandomInputArrays: () => S,
    createNetwork: () => tf.LayersModel,
    onNetworkReady?: (network: tf.LayersModel) => void,
}) {
    const trainValue = (network: tf.LayersModel, batch: LearnData<S>) => {
        const iteration = getNetworkExpIteration(network);
        const rb = new ReplayBuffer(batch.states.length);
        const mbs = config.miniBatchSize(iteration);
        const mbc = ceil(batch.size / mbs);

        console.info(`[Train Value]: Starting...
             Iteration ${iteration},
             Sum batch size: ${batch.size},
             Mini batch count: ${mbc} by ${mbs}`);

        const valueLossList: tf.Tensor[] = [];

        for (let i = 0; i < config.valueEpochs(iteration); i++) {
            for (let j = 0; j < mbc; j++) {
                const indices = rb.getSample(mbs, j * mbs, (j + 1) * mbs);
                const mBatch = createValueBatch(batch, indices);

                const tStates = createInputTensors(mBatch.states);
                const tReturns = tf.tensor1d(mBatch.returns);
                const tValues = tf.tensor1d(mBatch.values);

                const loss = trainValueNetwork(
                    network,
                    tStates,
                    tReturns,
                    tValues,
                    config.valueClipRatio, config.valueLossCoeff, config.clipNorm,
                    j === mbc - 1,
                );
                loss && valueLossList.push(loss);

                tf.dispose(tStates);
                tReturns.dispose();
                tValues.dispose();
            }
        }

        return onReadyRead()
            .then(() =>
                Promise.all(
                    valueLossList.map((t) => asyncUnwrapTensor(t).then((v) => v[0])),
                ),
            )
            .then((valueLossList) => {
                console.info(`[Train Value]: Finish`);

                if (valueLossList.some((v) => isLossDangerous(v, 1000))) {
                    throw new Error(`Value loss too dangerous: ${min(...valueLossList)} ${max(...valueLossList)}`);
                }

                metricsChannels.valueLoss.postMessage(valueLossList);
            });
    };

    const createValueBatch = (batch: LearnData<S>, indices: number[]) => {
        const states = indices.map(i => batch.states[i]);
        const values = indices.map(i => batch.values[i]);
        const returns = indices.map(i => batch.returns[i]);

        return {
            states: states,
            values: (values),
            returns: (returns),
        };
    };

    return createLearnerAgent<S>({
        config,
        createInputTensors,
        prepareRandomInputArrays,
        modelName: Model.Value,
        createNetwork,
        trainNetwork: trainValue,
        onNetworkReady,
    });
}
