import * as tf from '@tensorflow/tfjs';
import { onReadyRead } from '../Common/Tensor.ts';
import { getNetworkVersion } from '../Common/utils.ts';
import { LAST_NETWORK_VERSION, Model } from './def.ts';
import { createSupabaseIOHandler } from './supabaseStorage.ts';
import { shouldSaveHistoricalVersion } from './Utils.ts';

// Node.js file system paths for models
import { mkdirSync } from 'fs';
import { resolve } from 'path';
const MODELS_DIR = process.env.MODELS_DIR || './models';

function getModelSavePath(name: Model, version: number) {
    const absPath = resolve(MODELS_DIR, `v${version}-${name}`);
    try { mkdirSync(absPath, { recursive: true }); } catch { }
    return `file://${absPath}`;
}

function getModelLoadPath(name: Model, version: number) {
    const absPath = resolve(MODELS_DIR, `v${version}-${name}`, 'model.json');
    return `file://${absPath}`;
}

export async function saveNetwork(network: tf.LayersModel, name: Model) {
    await onReadyRead();

    const networkVersion = getNetworkVersion(network);
    const shouldSaveHistorical = await shouldSaveHistoricalVersion(name, networkVersion);

    // Save historical version
    if (shouldSaveHistorical) {
        console.info('Saving historical version of network:', name, networkVersion);
        // 1. Save to local file system
        await network.save(getModelSavePath(name, networkVersion), { includeOptimizer: true });
        // 2. Save to Supabase using IOHandler
        await network.save(createSupabaseIOHandler(name, networkVersion), { includeOptimizer: true });
    }

    // Save latest version
    // 1. Save to local file system
    await network.save(getModelSavePath(name, LAST_NETWORK_VERSION), { includeOptimizer: true });
    // 2. Save to Supabase using IOHandler
    await network.save(createSupabaseIOHandler(name, LAST_NETWORK_VERSION), { includeOptimizer: true });
}

export function loadNetwork(name: Model, version: number) {
    return tf.loadLayersModel(getModelLoadPath(name, version));
}

export async function loadLastNetwork(name: Model) {
    return loadNetwork(name, LAST_NETWORK_VERSION);
}

export async function loadNetworkByPath(path: string, name: Model) {
    const modelPath = `file://${path}/${name}.json`;
    const model = await tf.loadLayersModel(modelPath);
    return model;
}

