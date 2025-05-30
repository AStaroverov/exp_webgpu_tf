import { saveNetworkToDB } from '../../Models/Transfer.ts';
import * as tf from '@tensorflow/tfjs';
import { learningRateChannel, learnProcessChannel } from '../channels.ts';
import { networkHealthCheck } from '../train.ts';
import { get } from 'lodash';
import { getNetworkVersion, patientAction } from '../../Common/utils.ts';
import { LearnData } from './createLearnerManager.ts';
import { getModelFromDB } from '../../Models/Utils.ts';
import { Model } from '../../Models/def.ts';

export async function createLearnerAgent({ modelName, createNetwork, trainNetwork }: {
    modelName: Model,
    createNetwork: () => tf.LayersModel,
    trainNetwork: (network: tf.LayersModel, batch: LearnData) => unknown | Promise<unknown>,
}) {
    let network = await getModelFromDB(modelName, () => {
        const newNetwork = createNetwork();
        patientAction(() => saveNetworkToDB(newNetwork, modelName));
        return newNetwork;
    });

    learningRateChannel.obs.subscribe((lr) => {
        setLR(network, lr);
    });

    learnProcessChannel.response(async (batch: LearnData) => {
        try {
            await trainNetwork(network, batch);
            await patientAction(() => networkHealthCheck(network));
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
