import { isFunction } from 'lodash-es';
import { LAST_NETWORK_VERSION, Model } from '../../../ml-backend/src/Models/def.ts';
import * as tf from '../../../ml-common/tf';
import { patientAction } from '../../../ml-common/utils.ts';
import { loadModel } from './loader.ts';
import { getAvailableVersions } from './supabaseStorage.ts';


export async function getNetwork(modelName: Model, getInitial?: () => tf.LayersModel) {
    let network: undefined | tf.LayersModel;

    try {
        network = await patientAction(() => loadModel(modelName, LAST_NETWORK_VERSION), isFunction(getInitial) ? 1 : 10);
    } catch (error) {
        console.warn(`[getNetwork] Could not load model ${modelName} from Supabase:`, error);
        network = getInitial?.();
    }

    if (!network) {
        throw new Error(`Failed to load model ${modelName}`);
    }

    return network;
}

export async function getRandomHistoricalNetwork(modelName: Model): Promise<tf.LayersModel> {
    const versions = await getAvailableVersions(modelName);

    // Filter out latest version (v0) to get only historical versions
    const historicalVersions = versions.filter(v => v !== LAST_NETWORK_VERSION);

    if (historicalVersions.length === 0) {
        console.warn('[getRandomHistoricalNetwork] No historical versions found, using latest');
        return loadModel(modelName, LAST_NETWORK_VERSION);
    }

    // Pick random historical version
    const randomIndex = Math.floor(Math.random() * historicalVersions.length);
    const randomVersion = historicalVersions[randomIndex];

    console.info(`[getRandomHistoricalNetwork] Loading ${modelName} v${randomVersion} (from ${historicalVersions.length} historical versions)`);
    return loadModel(modelName, randomVersion);
}
