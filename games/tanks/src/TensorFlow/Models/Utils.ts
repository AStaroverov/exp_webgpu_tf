import { Config } from '../PPO/config.ts';
import { LayersModel } from '@tensorflow/tfjs';

export function getStorePath(name: string, config: Config): string {
    return `${ config.savePath }-${ name }`;
}

export function getIndexedDBModelPath(name: string, config: Config): string {
    return `indexeddb://${ getStorePath(name, config) }`;
}

export function disposeNetwork(network: LayersModel) {
    network.optimizer?.dispose();
    network.dispose();
}