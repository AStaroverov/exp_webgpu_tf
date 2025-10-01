import * as tf from '@tensorflow/tfjs';
import { LayersModel } from '@tensorflow/tfjs';
import { existsSync, readdirSync, statSync } from 'fs';
import { isFunction } from 'lodash-es';
import { resolve } from 'path';
import { random } from '../../../../lib/random.ts';
import { patientAction } from '../Common/utils.ts';
import { LAST_NETWORK_VERSION, Model, NetworkInfo } from './def.ts';
import { loadLastNetwork } from './Transfer.ts';

const MODELS_DIR = process.env.MODELS_DIR || './models';

export function disposeNetwork(network: LayersModel) {
    network.optimizer?.dispose();
    network.dispose();
}

export function getVersionFromStorePath(path: string): number {
    const splitted = path.split('|version:');
    return splitted.length == 2 ? Number(splitted[1]) : LAST_NETWORK_VERSION;
}

export async function getNetwork(modelName: Model, getInitial?: () => tf.LayersModel) {
    let network: undefined | tf.LayersModel;

    try {
        network = await patientAction(() => loadLastNetwork(modelName), isFunction(getInitial) ? 1 : 10);
    } catch (error) {
        console.warn(`[getNetwork] Could not load model ${modelName} from DB:`, error);
        network = getInitial?.();
    }

    if (!network) {
        throw new Error(`Failed to load model ${modelName}`);
    }

    return network;
}

/**
 * Get list of saved model versions from filesystem
 */
export async function getNetworkInfoList(model: Model): Promise<NetworkInfo[]> {
    const networkInfoList: NetworkInfo[] = [];

    if (!existsSync(MODELS_DIR)) {
        return networkInfoList;
    }

    try {
        const entries = readdirSync(MODELS_DIR, { withFileTypes: true });

        for (const entry of entries) {
            if (!entry.isDirectory()) {
                continue;
            }

            // Parse v{version}-{model} format
            const match = entry.name.match(/^v(\d+)-(.+)$/);
            if (!match) {
                continue;
            }

            const [, versionStr, modelName] = match;
            const version = parseInt(versionStr);

            // Filter by model name (e.g., 'policy-model' or 'value-model')
            if (isNaN(version) || modelName !== model) {
                continue;
            }

            const modelPath = resolve(MODELS_DIR, entry.name, 'model.json');
            if (!existsSync(modelPath)) {
                continue;
            }

            const stats = statSync(modelPath);
            networkInfoList.push({
                name: `v${version}`,
                path: modelPath,
                dateSaved: stats.mtime,
            });
        }

        // Sort by date descending (newest first)
        networkInfoList.sort((a, b) => b.dateSaved.getTime() - a.dateSaved.getTime());

        return networkInfoList;
    } catch (error) {
        console.error('Error reading model directories:', error);
        return [];
    }
}

/**
 * Get the second newest (penultimate) network version
 */
export async function getPenultimateNetworkVersion(name: Model): Promise<number | undefined> {
    const networkList = await getNetworkInfoList(name);

    // Filter out LAST_NETWORK_VERSION (v0) and get the second item
    const historicalVersions = networkList.filter((info) => {
        const version = parseInt(info.name.slice(1));
        return version !== LAST_NETWORK_VERSION;
    });

    if (historicalVersions.length === 0) {
        return undefined;
    }

    // Already sorted by date descending, so first item is the latest historical version
    const version = parseInt(historicalVersions[0].name.slice(1));
    return isNaN(version) ? undefined : version;
}

export async function shouldSaveHistoricalVersion(name: Model, version: number): Promise<boolean> {
    const step = 100_000;
    const penultimateNetworkVersion = await getPenultimateNetworkVersion(name);

    if (penultimateNetworkVersion == null) {
        return version > step;
    }

    if (Number.isNaN(penultimateNetworkVersion)) {
        console.error('penultimateNetworkVersion is NaN');
        return random() > 0.9;
    }

    return (version - penultimateNetworkVersion) > step;
}