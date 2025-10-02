import * as tf from '@tensorflow/tfjs';
import { LayersModel } from '@tensorflow/tfjs';
import { isFunction } from 'lodash-es';
import { isBrowser } from '../../../../lib/detect.ts';
import { random, randomRangeInt } from '../../../../lib/random.ts';
import { CONFIG } from '../../../ml-common/config.ts';
import { patientAction } from '../../../ml-common/utils.ts';
import { LAST_NETWORK_VERSION, Model, NetworkInfo } from './def.ts';
import { loadLastNetworkFromDB, loadNetworkFromDB } from './Transfer.ts';

export function disposeNetwork(network: LayersModel) {
    network.optimizer?.dispose();
    network.dispose();
}

export function getStorePath(name: string, version: number): string {
    const postfix = version === LAST_NETWORK_VERSION ? '' : `|version:${version}`;
    return `${CONFIG.savePath}-${name}${postfix}`;
}

export function getVersionFromStorePath(path: string): number {
    const splitted = path.split('|version:');
    return splitted.length == 2 ? Number(splitted[1]) : LAST_NETWORK_VERSION;
}

export function getIndexedDBModelPath(name: string, version: number): string {
    return `indexeddb://${getStorePath(name, version)}`;
}

export async function getNetwork(modelName: Model, getInitial?: () => tf.LayersModel) {
    let network: undefined | tf.LayersModel;

    try {
        network = await patientAction(() => loadLastNetworkFromDB(modelName), isFunction(getInitial) ? 1 : 10);
    } catch (error) {
        console.warn(`[getNetwork] Could not load model ${modelName} from DB:`, error);
        network = getInitial?.();
    }

    if (!network) {
        throw new Error(`Failed to load model ${modelName}`);
    }

    return network;
}

export async function getRandomHistoricalNetwork(modelName: Model) {
    const randomInfo = await getRandomNetworkInfo(modelName);
    let version = getVersionFromStorePath(randomInfo.name);
    version = Number.isNaN(version) ? LAST_NETWORK_VERSION : version;
    return patientAction(() => loadNetworkFromDB(modelName, version), 10);
}

const defaultSubNames = {
    [Model.Policy]: getStorePath(Model.Policy, LAST_NETWORK_VERSION),
    [Model.Value]: getStorePath(Model.Value, LAST_NETWORK_VERSION),
};

export async function getNetworkInfoList(model: Model) {
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
    const list = await getNetworkInfoList(model);
    return list[randomRangeInt(0, list.length - 1)];
}

export async function getPenultimateNetworkVersion(name: Model): Promise<number | undefined> {
    const defaultSubName = defaultSubNames[name];
    const penultimate = (await getNetworkInfoList(name)).reduce((acc, info) => {
        if (info.name !== defaultSubName && (acc === undefined || acc.dateSaved < info.dateSaved)) {
            return info;
        }
        return acc;
    }, undefined as undefined | NetworkInfo);

    return penultimate === undefined ? undefined : getVersionFromStorePath(penultimate.name);
}

export async function shouldSaveHistoricalVersion(name: Model, version: number) {
    const step = 500_000;
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
    const list = await getNetworkInfoList(name);

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