import { Config } from '../PPO/config.ts';
import * as tf from '@tensorflow/tfjs';
import { LayersModel } from '@tensorflow/tfjs';
import { loadNetworkFromDB, Model } from './Transfer.ts';

export function getStorePath(name: string, config: Config): string {
    return `${ config.savePath }-${ name }`;
}

export function getIndexedDBModelPath(name: string, config: Config): string {
    return `indexeddb://${ getStorePath(name, config) }`;
}


export async function getNetwork(modelName: Model, getInitial?: () => tf.LayersModel) {
    let network = await loadNetworkFromDB(modelName);

    if (!network) {
        if (!getInitial) {
            throw new Error('No network found and no initial network provided');
        }

        network = getInitial();
        console.log('Created a new model');
    } else {
        console.log('Model loaded successfully');
    }

    return network;
}

export function disposeNetwork(network: LayersModel) {
    network.optimizer?.dispose();
    network.dispose();
}