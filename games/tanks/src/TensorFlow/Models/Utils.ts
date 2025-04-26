import { Config } from '../PPO/config.ts';
import * as tf from '@tensorflow/tfjs';
import { LayersModel } from '@tensorflow/tfjs';
import { loadNetworkFromDB, Model } from './Transfer.ts';
import { patientAction } from '../Common/utils.ts';

export function getStorePath(name: string, config: Config): string {
    return `${ config.savePath }-${ name }`;
}

export function getIndexedDBModelPath(name: string, config: Config): string {
    return `indexeddb://${ getStorePath(name, config) }`;
}


export async function getNetwork(modelName: Model, getInitial: () => tf.LayersModel) {
    let network: undefined | tf.LayersModel;

    try {
        network = await patientAction(() => loadNetworkFromDB(modelName), 3);
    } catch (error) {
        console.warn(`[getNetwork] Could not load model ${ modelName } from DB:`, error);
        network = getInitial?.();
    }

    return network;
}

export function disposeNetwork(network: LayersModel) {
    network.optimizer?.dispose();
    network.dispose();
}