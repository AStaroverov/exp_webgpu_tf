import * as tf from '@tensorflow/tfjs';
import { get } from 'lodash';
import { getNetworkExpIteration, patientAction, setNetworkExpIteration, setNetworkLearningRate } from '../../../../ml-common/utils.ts';
import { saveNetworkToDB } from '../../Models/Transfer.ts';
import { getNetwork } from '../../Models/Utils.ts';
import { Model } from '../../Models/def.ts';
import { learnProcessChannel, modelSettingsChannel } from '../channels.ts';
import { networkHealthCheck } from '../train.ts';
import { LearnData } from './createLearnerManager.ts';

export async function createLearnerAgent({ modelName, createNetwork, trainNetwork }: {
    modelName: Model,
    createNetwork: () => tf.LayersModel,
    trainNetwork: (network: tf.LayersModel, batch: LearnData) => unknown | Promise<unknown>,
}) {
    let network = await getNetwork(modelName, () => {
        const newNetwork = createNetwork();
        patientAction(() => saveNetworkToDB(newNetwork, modelName));
        return newNetwork;
    });

    modelSettingsChannel.obs.subscribe(({ lr, expIteration }) => {
        lr && setNetworkLearningRate(network, lr);
        expIteration && setNetworkExpIteration(network, expIteration);
    });

    learnProcessChannel.response(async (batch: LearnData) => {
        try {
            await trainNetwork(network, batch);
            await patientAction(() => networkHealthCheck(network));
            await patientAction(() => saveNetworkToDB(network, modelName));

            return { modelName: modelName, version: getNetworkExpIteration(network) };
        } catch (e) {
            console.error(e);
            return { modelName: modelName, error: get(e, 'message') ?? 'Unknown error' };
        }
    });
}
