import * as tf from '@tensorflow/tfjs';
import { onReadyRead } from '../Common/Tensor.ts';
import { getNetworkVersion } from '../Common/utils.ts';
import { LAST_NETWORK_VERSION, Model } from './def.ts';
import { shouldSaveHistoricalVersion } from './Utils.ts';

// Node.js file system paths for models
import { mkdirSync } from 'fs';
import { resolve } from 'path';
const MODELS_DIR = process.env.MODELS_DIR || './models';

function getNodeModelSavePath(version: number) {
    const absPath = resolve(MODELS_DIR, `v${version}`);
    try {
        mkdirSync(absPath, { recursive: true });
    } catch (e) {
        // Directory might already exist
    }
    return `file://${absPath}`;
}

function getNodeModelLoadPath(name: Model, version: number) {
    // For loading: models/v{version}/{model-name}.json
    const absPath = resolve(MODELS_DIR, `v${version}/${name}.json`);
    return `file://${absPath}`;
}

export async function saveNetworkToDB(network: tf.LayersModel, name: Model) {
    await onReadyRead();

    const networkVersion = getNetworkVersion(network);
    const shouldSaveHistorical = await shouldSaveHistoricalVersion(name, networkVersion);

    if (shouldSaveHistorical) {
        console.info('Saving historical version of network:', name, networkVersion);
        await network.save(getNodeModelSavePath(networkVersion), { includeOptimizer: true });
    }

    return network.save(getNodeModelSavePath(LAST_NETWORK_VERSION), { includeOptimizer: true });
}

export function loadNetworkFromDB(name: Model, version: number) {
    return tf.loadLayersModel(getNodeModelLoadPath(name, version));
}

export async function loadLastNetworkFromDB(name: Model) {
    return loadNetworkFromDB(name, LAST_NETWORK_VERSION);
}

export async function loadNetworkFromFS(path: string, name: Model) {
    const modelPath = path.startsWith('file://') ? path : `file://${path}/${name}.json`;
    const model = await tf.loadLayersModel(modelPath);
    return model;
}

export function downloadNetwork(name: Model) {
    return loadNetworkFromDB(name, LAST_NETWORK_VERSION)
        .then((network) => network.save(`downloads://${name}`, { includeOptimizer: true }));
}
