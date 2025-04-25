import { loadNetworkFromDB, Model, saveNetworkToDB } from '../../Models/Transfer.ts';
import * as tf from '@tensorflow/tfjs';
import { FinalBatch } from './LearnerAgent.ts';
import { learningRateChannel, learnMemoryChannel } from '../channels.ts';
import { networkHealthCheck } from '../train.ts';
import { get } from 'lodash';
import { setModelState } from '../../Common/modelsCopy.ts';
import { getNetworkVersion, patientAction } from '../../Common/utils.ts';

export function createLearnerAgent({ modelName, createNetwork, trainNetwork }: {
    modelName: Model,
    createNetwork: () => tf.LayersModel,
    trainNetwork: (network: tf.LayersModel, batch: FinalBatch) => void,
}) {
    const network = createNetwork();
    const loaded = upsertNetwork(modelName, network);

    learningRateChannel.obs.subscribe((lr) => {
        setLR(network, lr);
    });

    learnMemoryChannel.response(async (batch) => {
        try {
            await loaded;

            trainNetwork(network, batch);

            const healthy = await patientAction(() => networkHealthCheck(network));

            if (!healthy) {
                throw new Error('Health check failed');
            }

            await patientAction(() => saveNetwork(modelName, network));

            return { modelName: modelName, version: getNetworkVersion(network) };
        } catch (e) {
            return { modelName: modelName, error: get(e, 'message') ?? 'Unknown error' };
        }
    });
}

function setLR(o: tf.LayersModel, lr: number) {
    // @ts-expect-error
    o.optimizer.learningRate = lr;
}

export async function upsertNetwork(modelName: Model, outNetwork: tf.LayersModel) {
    try {
        const network = await loadNetworkFromDB(modelName);

        if (!network) return false;

        await setModelState(outNetwork, network);

        network.dispose();

        console.log('Models loaded successfully');
        return true;
    } catch (error) {
        console.warn('Could not load models, starting with new ones:', error);
        return false;
    }
}

async function saveNetwork(modelName: Model, network: tf.LayersModel) {
    try {
        await saveNetworkToDB(network, modelName);
        return true;
    } catch (error) {
        console.error('Error saving models:', error);
        return false;
    }
}