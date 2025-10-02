import * as tf from '@tensorflow/tfjs';
import { isBrowser } from '../../../../lib/detect.ts';
import { onReadyRead } from '../Common/Tensor.ts';
import { getNetworkExpIteration } from '../Common/utils.ts';
import { LAST_NETWORK_VERSION, Model } from './def.ts';
import { getIndexedDBModelPath, removeOutLimitNetworks, shouldSaveHistoricalVersion } from './Utils.ts';

export async function saveNetworkToDB(network: tf.LayersModel, name: Model) {
    await onReadyRead();

    const networkVersion = getNetworkExpIteration(network);
    const shouldSaveHistorical = await shouldSaveHistoricalVersion(name, networkVersion);

    if (isBrowser) {
        if (shouldSaveHistorical) {
            console.info('Saving historical version of network:', name, networkVersion);
            void network.save(getIndexedDBModelPath(name, networkVersion), { includeOptimizer: true });
            void removeOutLimitNetworks(name);
        }

        return network.save(getIndexedDBModelPath(name, LAST_NETWORK_VERSION), { includeOptimizer: true });
    }

    throw new Error('Unsupported environment for saving model');
}

export function loadNetworkFromDB(name: Model, version: number) {
    if (isBrowser) {
        return tf.loadLayersModel(getIndexedDBModelPath(name, version));
    }

    throw new Error('Unsupported environment for loading model');
}

export async function loadLastNetworkFromDB(name: Model) {
    return loadNetworkFromDB(name, LAST_NETWORK_VERSION);
}

export async function loadNetworkFromFS(path: string, name: Model) {
    if (isBrowser) {
        const modelPath = `${path}/${name}.json`;
        const model = await tf.loadLayersModel(modelPath);

        return model;
    }

    throw new Error('Unsupported environment for loading model');
}

export function downloadNetwork(name: Model) {
    return loadNetworkFromDB(name, LAST_NETWORK_VERSION)
        .then((network) => network.save(`downloads://${name}`, { includeOptimizer: true }));
}
