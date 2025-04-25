import { Model } from '../../Models/Transfer.ts';
import { createLearnerAgent } from './createLearnerAgent.ts';
import { createValueNetwork } from '../../Models/Create.ts';
import { FinalBatch } from './LearnerAgent.ts';
import { CONFIG } from '../config.ts';
import * as tf from '@tensorflow/tfjs';
import { trainValueNetwork } from '../train.ts';
import { createInputTensors } from '../../Common/InputTensors.ts';
import { ReplayBuffer } from '../../Common/ReplayBuffer.ts';
import { ceil } from '../../../../../../lib/math.ts';
import { metricsChannels } from '../../Common/channels.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { getNetworkVersion } from '../../Common/utils.ts';

export function createValueLearnerAgent() {
    createLearnerAgent({
        modelName: Model.Value,
        createNetwork: createValueNetwork,
        trainNetwork: trainValue,
    });
}

function trainValue(network: tf.LayersModel, batch: FinalBatch) {
    const rb = new ReplayBuffer(batch.states.length);
    const mbs = CONFIG.miniBatchSize;
    const mbc = ceil(batch.size / mbs);
    const version = getNetworkVersion(network);

    console.log(`[Train Policy]: Iteration ${ version },
         Sum batch size: ${ batch.size },
         Mini batch count: ${ mbc } by ${ mbs }`);

    const valueLossList: number[] = [];

    for (let i = 0; i < CONFIG.valueEpochs; i++) {
        for (let j = 0; j < mbc; j++) {
            const indices = rb.getSample(mbs, j * mbs, (j + 1) * mbs);
            const mBatch = createValueBatch(batch, indices);

            tf.tidy(() => {
                const loss = trainValueNetwork(
                    network,
                    createInputTensors(mBatch.states),
                    tf.tensor1d(mBatch.returns),
                    tf.tensor1d(mBatch.values),
                    CONFIG.valueClipRatio, CONFIG.valueLossCoeff, CONFIG.clipNorm,
                    j === mbc - 1,
                );

                loss && valueLossList.push(loss);
            });
        }
    }

    macroTasks.addTimeout(() => {
        metricsChannels.valueLoss.postMessage(valueLossList);
    }, 0);
}

function createValueBatch(batch: FinalBatch, indices: number[]) {
    const states = indices.map(i => batch.states[i]);
    const values = indices.map(i => batch.values[i]);
    const returns = indices.map(i => batch.returns[i]);

    return {
        states: states,
        values: (values),
        returns: (returns),
    };
}
