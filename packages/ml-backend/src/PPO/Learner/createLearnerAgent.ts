import * as tf from '@tensorflow/tfjs';
import { get } from 'lodash-es';
import { getNetworkVersion, patientAction } from '../../Common/utils.ts';
import { saveNetwork } from '../../Models/Transfer.ts';
import { getNetwork } from '../../Models/Utils.ts';
import { Model } from '../../Models/def.ts';
import { learningRateChannel, learnProcessChannel } from '../localChannels.ts';
import { networkHealthCheck } from '../train.ts';
import { LearnData } from './createLearnerManager.ts';

export async function createLearnerAgent({ modelName, createNetwork, trainNetwork }: {
    modelName: Model,
    createNetwork: () => tf.LayersModel,
    trainNetwork: (network: tf.LayersModel, batch: LearnData) => unknown | Promise<unknown>,
}) {
    let network = await getNetwork(modelName, () => {
        const newNetwork = createNetwork();
        patientAction(() => saveNetwork(newNetwork, modelName));
        return newNetwork;
    });

    learningRateChannel.obs.subscribe((lr) => {
        setLR(network, lr);
    });

    learnProcessChannel.response(async (batch: LearnData) => {
        try {
            await trainNetwork(network, batch);
            await patientAction(() => networkHealthCheck(network));
            await patientAction(() => saveNetwork(network, modelName));

            const version = getNetworkVersion(network);
            console.info(`✅ ${modelName} v${version} trained and saved (Supabase sync included)`);

            return { modelName: modelName, version };
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
