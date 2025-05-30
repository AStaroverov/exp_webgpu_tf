import { CONFIG } from '../PPO/config.ts';
import * as tf from '@tensorflow/tfjs';
import { LayersModel } from '@tensorflow/tfjs';
import { loadLastModelFromDB, loadModelFromDB, loadModelFromFS } from './Transfer.ts';
import { patientAction } from '../Common/utils.ts';
import { isFunction } from 'lodash-es';
import { random, randomRangeInt } from '../../../../../lib/random.ts';
import { LAST_NETWORK_VERSION, Model, NetworkInfo } from './def.ts';
import { isBrowser } from '../../../../../lib/detect.ts';

export function disposeNetwork(network: LayersModel) {
    network.optimizer?.dispose();
    network.dispose();
}

export function getStorePath(name: string, version: number): string {
    const postfix = version === LAST_NETWORK_VERSION ? '' : `|version:${ version }`;
    return `${ CONFIG.savePath }-${ name }${ postfix }`;
}

export function getVersionFromStorePath(path: string): number {
    const splitted = path.split('|version:');
    return splitted.length == 2 ? Number(splitted[1]) : LAST_NETWORK_VERSION;
}

export function getIndexedDBModelPath(name: string, version: number): string {
    return `indexeddb://${ getStorePath(name, version) }`;
}

export async function getModelFromDB(modelName: Model, getInitial?: () => tf.LayersModel) {
    let model: undefined | tf.LayersModel;

    try {
        model = await patientAction(() => loadLastModelFromDB(modelName), isFunction(getInitial) ? 1 : 10);
    } catch (error) {
        console.warn(`[getModelFromDB] Could not load model ${ modelName } from DB:`, error);
        model = getInitial?.();
    }

    if (!model) {
        throw new Error(`Failed to load model ${ modelName }`);
    }

    return model;
}

export async function getModelFromFS(modelName: Model, path: string) {
    return patientAction(() => loadModelFromFS(path, modelName), 10);
}

export async function getRandomHistoricalModel(modelName: Model) {
    const randomInfo = await getRandomNetworkInfo(modelName);
    let version = getVersionFromStorePath(randomInfo.name);
    version = Number.isNaN(version) ? LAST_NETWORK_VERSION : version;
    return patientAction(() => loadModelFromDB(modelName, version), 10);
}

const defaultSubNames = {
    [Model.Policy]: getStorePath(Model.Policy, LAST_NETWORK_VERSION),
    [Model.Value]: getStorePath(Model.Value, LAST_NETWORK_VERSION),
};

export async function getModelsInfoList(model: Model) {
    return tf.io.listModels().then((v) => Object.entries(v).reduce((acc, [name, info]) => {
        if (name.includes(defaultSubNames[model])) {
            acc.push({
                name: name.split('://')[1],
                path: name,
                dateSaved: info.dateSaved,
            });
        }
        return acc;
    }, [] as NetworkInfo[]));
}

export async function getRandomNetworkInfo(model: Model) {
    const list = await getModelsInfoList(model);
    return list[randomRangeInt(0, list.length - 1)];
}

export async function getPenultimateNetworkVersion(name: Model): Promise<number | undefined> {
    const defaultSubName = defaultSubNames[name];
    const penultimate = (await getModelsInfoList(name)).reduce((acc, info) => {
        if (info.name !== defaultSubName && (acc === undefined || acc.dateSaved < info.dateSaved)) {
            return info;
        }
        return acc;
    }, undefined as undefined | NetworkInfo);

    return penultimate === undefined ? undefined : getVersionFromStorePath(penultimate.name);
}

export async function shouldSaveHistoricalVersion(name: Model, version: number) {
    const step = 100_000; // 100_000 / epoch * batch/mini_batch == 625 learns steps
    const penultimateNetworkVersion = await getPenultimateNetworkVersion(name);

    if (penultimateNetworkVersion == null) {
        return version > step;
    }

    if (Number.isNaN(penultimateNetworkVersion)) {
        console.error('penultimateNetworkVersion is NaN');
        return random() > 0.9;
    }

    return version - penultimateNetworkVersion > step;
}

const NETWORKS_LIMIT_COUNT = 20;

export async function removeOutLimitNetworks(name: Model) {
    const list = await getModelsInfoList(name);

    if (list.length > NETWORKS_LIMIT_COUNT) {
        const forRemove = list
            .sort((a, b) => a.dateSaved.getTime() - b.dateSaved.getTime())
            .slice(0, list.length - NETWORKS_LIMIT_COUNT);


        if (isBrowser) {
            for (const networkInfo of forRemove) {
                tf.io.removeModel(networkInfo.path);
            }
        } else {
            throw new Error('Unsupported environment for removing model');
        }
    }
}