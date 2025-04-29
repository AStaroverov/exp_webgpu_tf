import { getIndexedDBModelPath } from './Utils.ts';
import { CONFIG } from '../PPO/config.ts';
import * as tf from '@tensorflow/tfjs';
import { isBrowser } from '../../../../../lib/detect.ts';
import { onReadyRead } from '../Common/Tensor.ts';

export enum Model {
    Policy = 'policy-model',
    Value = 'value-model',
}

export async function saveNetworkToDB(network: tf.LayersModel, name: Model) {
    await onReadyRead();

    if (isBrowser) {
        return network.save(getIndexedDBModelPath(name, CONFIG), { includeOptimizer: true });
    }

    throw new Error('Unsupported environment for saving model');
}

export function loadNetworkFromDB(name: Model) {
    if (isBrowser) {
        return tf.loadLayersModel(getIndexedDBModelPath(name, CONFIG));
    }

    throw new Error('Unsupported environment for loading model');
}

export async function loadNetworkFromFS(path: string, name: Model) {
    if (isBrowser) {
        const modelPath = `src/TensorFlow/Models/Trained/${ path }/${ name }.json`;
        const model = await tf.loadLayersModel(modelPath);

        return model;
    }

    throw new Error('Unsupported environment for loading model');
}

export function downloadNetwork(name: Model) {
    return loadNetworkFromDB(name).then((network) => network.save(`downloads://${ name }`, { includeOptimizer: true }));
}
