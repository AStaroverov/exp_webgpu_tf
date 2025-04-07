import { getStoreModelPath } from './Utils.ts';
import { CONFIG } from '../PPO/config.ts';
import * as tf from '@tensorflow/tfjs';

export enum Model {
    Policy = 'policy-model',
    Value = 'value-model',
}

export function saveNetwork(network: tf.LayersModel, name: Model) {
    return network.save(getStoreModelPath(name, CONFIG), { includeOptimizer: true });
}

export function loadNetwork(name: Model) {
    return tf.loadLayersModel(getStoreModelPath(name, CONFIG));
}

export function downloadNetwork(name: Model) {
    return loadNetwork(name).then((network) => network.save(`downloads://${ name }`, { includeOptimizer: true }));
}
