import * as tf from '@tensorflow/tfjs';
import { get, isNumber } from 'lodash-es';
import { getNetworkExpIteration, patientAction, setNetworkExpIteration, setNetworkLearningRate, setNetworkPerturbConfig } from '../../../../ml-common/utils.ts';
import { saveNetworkToDB } from '../../Models/Transfer.ts';
import { disposeNetwork, getNetwork } from '../../Models/Utils.ts';
import { Model } from '../../Models/def.ts';
import { learnProcessChannel, modelSettingsChannel } from '../channels.ts';
import { LearnData } from './createLearnerManager.ts';

// Note: networkHealthCheck not needed for SAC (was PPO-specific)

export async function createLearnerAgent({ modelName, createNetwork, trainNetwork }: {
    modelName: Model,
    createNetwork: (...args: any[]) => tf.LayersModel,
    trainNetwork: (network: tf.LayersModel, batch: LearnData) => unknown | Promise<unknown>,
}) {
    let network = await getNetwork(modelName, () => {
        const newNetwork = createNetwork(modelName);
        patientAction(() => saveNetworkToDB(newNetwork, modelName));
        return newNetwork;
    });

    modelSettingsChannel.obs.subscribe(({ lr, perturbChance, perturbScale, expIteration }) => {
        isNumber(lr) && setNetworkLearningRate(network, lr);
        isNumber(expIteration) && setNetworkExpIteration(network, expIteration);
        (isNumber(perturbChance) && isNumber(perturbScale)) && setNetworkPerturbConfig(network, perturbChance, perturbScale);
    });

    learnProcessChannel.response(async (batch: LearnData) => {
        try {
            await trainNetwork(network, batch);
            // TODO: Add network health check for SAC if needed
            await patientAction(() => saveNetworkToDB(network, modelName));

            return { modelName: modelName, version: getNetworkExpIteration(network) };
        } catch (e: Error | unknown) {
            console.error(e);

            disposeNetwork(network);
            console.info('Load last network after error...');
            network = await patientAction(() => getNetwork(modelName));

            return { modelName: modelName, error: get(e, 'message') ?? 'Unknown error', restart: e instanceof Error && e.message.includes('mem') };
        }
    });
}
