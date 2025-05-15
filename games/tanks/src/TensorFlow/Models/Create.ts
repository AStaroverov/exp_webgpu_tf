import * as tf from '@tensorflow/tfjs';
import {
    ALLY_BUFFER,
    BULLET_BUFFER,
    ENEMY_BUFFER,
    MAX_ALLIES,
    MAX_BULLETS,
    MAX_ENEMIES,
} from '../../Game/ECS/Components/TankState.ts';
import { ACTION_DIM } from '../Common/consts.ts';
import { CONFIG } from '../PPO/config.ts';
import {
    applyCrossAttentionLayer,
    applyDenseLayers,
    applyEncoding,
    applyTransformLayers,
    convertInputsToTokens,
    createInputs,
} from './ApplyLayers.ts';

import { Model } from './def.ts';
import { PatchedAdamOptimizer } from './PatchedAdamOptimizer.ts';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';

export const CONTROLLER_FEATURES_DIM = 5;
export const BATTLE_FEATURES_DIM = 6;
export const TANK_FEATURES_DIM = 8;
export const ENEMY_SLOTS = MAX_ENEMIES;
export const ENEMY_FEATURES_DIM = ENEMY_BUFFER - 1; // -1 потому что id не считаем
export const ALLY_SLOTS = MAX_ALLIES;
export const ALLY_FEATURES_DIM = ALLY_BUFFER - 1; // -1 потому что id не считаем
export const BULLET_SLOTS = MAX_BULLETS;
export const BULLET_FEATURES_DIM = BULLET_BUFFER - 1; // -1 потому что id не считаем

type NetworkConfig = {
    dim: number;
    heads: number;
    dropout?: number;
    hiddenDenseLayers: [ActivationIdentifier, number][];
    flatDenseLayers: [ActivationIdentifier, number][];
}

const policyNetworkConfig: NetworkConfig = {
    dim: 64,
    heads: 4,
    hiddenDenseLayers: [
        ['relu', 64 * 2],
        ['relu', 64],
        ['relu', 32],
    ] as [ActivationIdentifier, number][],
    flatDenseLayers: [
        ['relu', 512],
        ['relu', 256],
        ['relu', 64],
    ] as [ActivationIdentifier, number][],
};
const valueNetworkConfig: NetworkConfig = {
    dim: 16,
    heads: 1,
    hiddenDenseLayers: [
        ['relu', 16 * 2],
        ['relu', 16],
    ] as [ActivationIdentifier, number][],
    flatDenseLayers: [
        ['relu', 128],
        ['relu', 16],
    ] as [ActivationIdentifier, number][],
};

export function createPolicyNetwork(): tf.LayersModel {
    const { inputs, network } = createBaseNetwork(Model.Policy, policyNetworkConfig);
    // Выход: ACTION_DIM * 2 (пример: mean и logStd) ---
    const policyOutput = tf.layers.dense({
        name: Model.Policy + '_output',
        units: ACTION_DIM * 2,
        activation: 'linear',
    }).apply(network) as tf.SymbolicTensor;
    const model = tf.model({
        name: Model.Policy,
        inputs: Object.values(inputs),
        outputs: policyOutput,
    });
    model.optimizer = new PatchedAdamOptimizer(CONFIG.lrConfig.initial);
    // fake loss for save optimizer with model
    model.loss = 'meanSquaredError';

    return model;
}

export function createValueNetwork(): tf.LayersModel {
    const { inputs, network } = createBaseNetwork(Model.Value, valueNetworkConfig);
    const valueOutput = tf.layers.dense({
        name: Model.Value + '_output',
        units: 1,
        activation: 'linear',
    }).apply(network) as tf.SymbolicTensor;
    const model = tf.model({
        name: Model.Value,
        inputs: Object.values(inputs),
        outputs: valueOutput,
    });
    model.optimizer = new PatchedAdamOptimizer(CONFIG.lrConfig.initial);
    // fake loss for save optimizer with model
    model.loss = 'meanSquaredError';

    return model;
}

function createBaseNetwork(modelName: Model, config: typeof policyNetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    const tankToEnemiesAttn = applyEncoding(
        applyCrossAttentionLayer(modelName + '_tankToEnemiesAttn', config.heads, {
            qTok: tokens.tankTok,
            kvTok: tokens.enemiesTok,
            kvMask: inputs.enemiesMaskInput,
        }),
    );
    const tankToAlliesAttn = applyEncoding(
        applyCrossAttentionLayer(modelName + '_tankToAlliesAttn', config.heads, {
            qTok: tokens.tankTok,
            kvTok: tokens.alliesTok,
            kvMask: inputs.alliesMaskInput,
        }),
    );
    const tankToBulletsAttn = applyEncoding(
        applyCrossAttentionLayer(modelName + '_tankToBulletsAttn', config.heads, {
            qTok: tokens.tankTok,
            kvTok: tokens.bulletsTok,
            kvMask: inputs.bulletsMaskInput,
        }),
    );

    const envToken = tf.layers.concatenate({ name: modelName + '_envToken', axis: 1 }).apply([
        tokens.controllerTok,
        tokens.battleTok,
        tokens.tankTok,
        tankToEnemiesAttn,
        tankToAlliesAttn,
        tankToBulletsAttn,
    ]) as tf.SymbolicTensor;

    const transformedEnvToken = applyTransformLayers(
        modelName + '_transformedEnvToken',
        {
            depth: 2,
            numHeads: config.heads,
            dropout: config.dropout,
            token: envToken,
        },
    );

    const normTransformedEnvToken = tf.layers.layerNormalization({ name: modelName + '_normTransformedEnvToken' }).apply(transformedEnvToken) as tf.SymbolicTensor;

    const hiddenMLP = applyDenseLayers(
        modelName + '_hiddenMLP',
        normTransformedEnvToken,
        config.hiddenDenseLayers,
    );

    const flattenFinalToken = tf.layers.flatten({ name: modelName + '_flattenFinalToken' }).apply(hiddenMLP) as tf.SymbolicTensor;
    const normFinalToken = tf.layers.layerNormalization({ name: modelName + '_normFinalToken' }).apply(flattenFinalToken) as tf.SymbolicTensor;

    const flattenMLP = applyDenseLayers(
        modelName + '_flattenMLP',
        normFinalToken,
        config.flatDenseLayers,
    );

    return { inputs, network: flattenMLP };
}
