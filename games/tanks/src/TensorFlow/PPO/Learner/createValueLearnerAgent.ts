import { Model } from '../../Models/Transfer.ts';
import { createLearnerAgent } from './createLearnerAgent.ts';
import { createValueNetwork } from '../../Models/Create.ts';
import { CONFIG } from '../config.ts';
import * as tf from '@tensorflow/tfjs';
import { trainValueNetwork } from '../train.ts';
import { createInputTensors } from '../../Common/InputTensors.ts';
import { ReplayBuffer } from '../../Common/ReplayBuffer.ts';
import { ceil, max, min } from '../../../../../../lib/math.ts';
import { metricsChannels } from '../../Common/channels.ts';
import { getNetworkVersion } from '../../Common/utils.ts';
import { LearnData } from './createLearnerManager.ts';
import { asyncUnwrapTensor, onReadyRead } from '../../Common/Tensor.ts';
import { isLossDangerous } from './isLossDangerous.ts';

export function createValueLearnerAgent() {
    return createLearnerAgent({
        modelName: Model.Value,
        createNetwork: createValueNetwork,
        trainNetwork: trainValue,
    });
}

function trainValue(network: tf.LayersModel, batch: LearnData) {
    const rb = new ReplayBuffer(batch.states.length);
    const mbs = CONFIG.miniBatchSize;
    const mbc = ceil(batch.size / mbs);
    const version = getNetworkVersion(network);

    console.log(`[Train Value]: Iteration ${ version },
         Sum batch size: ${ batch.size },
         Mini batch count: ${ mbc } by ${ mbs }`);

    const valueLossList: tf.Tensor[] = [];

    for (let i = 0; i < CONFIG.valueEpochs; i++) {
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
                CONFIG.valueClipRatio, CONFIG.valueLossCoeff, CONFIG.clipNorm,
                true,//j === mbc - 1,
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
            if (valueLossList.some((v) => isLossDangerous(v, 100))) {
                throw new Error(`Value loss too dangerous: ${ min(...valueLossList) } ${ max(...valueLossList) }`);
            }

            metricsChannels.valueLoss.postMessage(valueLossList);
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
