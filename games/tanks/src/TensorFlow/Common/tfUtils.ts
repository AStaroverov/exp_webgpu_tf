import { Config } from '../PPO/config.ts';

export function getStorePath(name: string, config: Config): string {
    return `${ config.savePath }-${ name }`;
}

export function getStoreModelPath(name: string, config: Config): string {
    return `indexeddb://${ getStorePath(name, config) }`;
}