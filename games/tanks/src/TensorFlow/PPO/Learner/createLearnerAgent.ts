import { Model, saveNetworkToDB } from '../../Models/Transfer.ts';
import * as tf from '@tensorflow/tfjs';
import { learningRateChannel, learnMemoryChannel } from '../channels.ts';
import { networkHealthCheck } from '../train.ts';
import { get } from 'lodash';
import { getNetworkVersion, patientAction } from '../../Common/utils.ts';
import { LearnBatch } from './createLearnerManager.ts';
import { getNetwork } from '../../Models/Utils.ts';

export async function createLearnerAgent({ modelName, createNetwork, trainNetwork }: {
    modelName: Model,
    createNetwork: () => tf.LayersModel,
    trainNetwork: (network: tf.LayersModel, batch: LearnBatch) => void,
}) {
    let network = await getNetwork(modelName, createNetwork);

    learningRateChannel.obs.subscribe((lr) => {
        setLR(network, lr);
    });

    learnMemoryChannel.response(async (batch: LearnBatch) => {
        try {
            trainNetwork(network, batch);

            const healthy = await patientAction(() => networkHealthCheck(network));

            if (!healthy) {
                throw new Error('Health check failed');
            }

            await patientAction(() => saveNetworkToDB(network, modelName));

            return { modelName: modelName, version: getNetworkVersion(network) };
        } catch (e) {
            console.error(e);
            return { modelName: modelName, error: get(e, 'message') ?? 'Unknown error' };
        }
    });
}

function setLR(o: tf.LayersModel, lr: number) {
    // @ts-expect-error
    o.optimizer.learningRate = lr;
}
