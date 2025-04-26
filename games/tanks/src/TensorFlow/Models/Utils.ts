import { Config } from '../PPO/config.ts';
import * as tf from '@tensorflow/tfjs';
import { LayersModel } from '@tensorflow/tfjs';
import { loadNetworkFromDB, Model } from './Transfer.ts';
import { patientAction } from '../Common/utils.ts';
import { isFunction } from 'lodash-es';

export function getStorePath(name: string, config: Config): string {
    return `${ config.savePath }-${ name }`;
}

export function getIndexedDBModelPath(name: string, config: Config): string {
    return `indexeddb://${ getStorePath(name, config) }`;
}

export async function getNetwork(modelName: Model, getInitial?: () => tf.LayersModel) {
    let network: undefined | tf.LayersModel;

    try {
        network = await patientAction(() => loadNetworkFromDB(modelName), isFunction(getInitial) ? 1 : 10);
    } catch (error) {
        console.warn(`[getNetwork] Could not load model ${ modelName } from DB:`, error);
        network = getInitial?.();
    }

    if (!network) {
        throw new Error(`Failed to load model ${ modelName }`);
    }

    return network;
}

export function disposeNetwork(network: LayersModel) {
    network.optimizer?.dispose();
    network.dispose();
}