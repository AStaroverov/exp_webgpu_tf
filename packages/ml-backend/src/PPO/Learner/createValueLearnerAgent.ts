import { ceil, max, min } from '../../../../../lib/math.ts';
import * as tf from '../../../../ml-common/tf';
// import { metricsChannels } from '../../Common/channels.ts'; // metrics disabled
import { createInputTensors } from '../../../../ml-common/InputTensors.ts';
import { ReplayBuffer } from '../../../../ml-common/ReplayBuffer.ts';
import { asyncUnwrapTensor, onReadyRead } from '../../../../ml-common/Tensor.ts';
import { getNetworkExpIteration } from '../../../../ml-common/utils.ts';
import { createValueNetwork } from '../../Models/Create.ts';
import { Model } from '../../Models/def.ts';
import { CONFIG } from '../config.ts';
import { trainValueNetwork } from '../train.ts';
import { createLearnerAgent } from './createLearnerAgent.ts';
import { LearnData } from './createLearnerManager.ts';
import { isLossDangerous } from './isLossDangerous.ts';

export function createValueLearnerAgent() {
    return createLearnerAgent({
        modelName: Model.Value,
        createNetwork: createValueNetwork,
        trainNetwork: trainValue,
    });
}

function trainValue(network: tf.LayersModel, batch: LearnData) {
    const version = getNetworkExpIteration(network);
    const rb = new ReplayBuffer(batch.states.length);
    const mbs = CONFIG.miniBatchSize(version);
    const mbc = ceil(batch.size / mbs);

    console.info(`[Train Value]: Starting...
         Iteration ${version},
         Sum batch size: ${batch.size},
         Mini batch count: ${mbc} by ${mbs}`);

    const valueLossList: tf.Tensor[] = [];

    for (let i = 0; i < CONFIG.valueEpochs(version); i++) {
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
                mbs,
                CONFIG.valueClipRatio, CONFIG.valueLossCoeff, CONFIG.clipNorm,
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

            if (valueLossList.some((v) => isLossDangerous(v, 2000))) {
                throw new Error(`Value loss too dangerous: ${min(...valueLossList)} ${max(...valueLossList)}`);
            }

            // metricsChannels.valueLoss.postMessage(valueLossList); // disabled
        });
}

function createValueBatch(batch: LearnData, indices: number[]) {
    const states = indices.map(i => batch.states[i]);
    const values = indices.map(i => batch.values[i]);
    const returns = indices.map(i => batch.returns[i]);

    return {
        states: states,
        values: (values),
        returns: (returns),
    };
}
