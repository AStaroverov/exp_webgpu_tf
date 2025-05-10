import * as tf from '@tensorflow/tfjs';
import {
    ALLY_BUFFER,
    BULLET_BUFFER,
    ENEMY_BUFFER,
    MAX_ALLIES,
    MAX_BULLETS,
    MAX_ENEMIES,
} from '../../ECS/Components/TankState.ts';
import { ACTION_DIM } from '../Common/consts.ts';
import { CONFIG } from '../PPO/config.ts';
import {
    applyCrossAttentionLayer,
    applyDenseLayers,
    applySelfAttentionLayer,
    convertInputsToTokens,
    createInputs,
} from './Layers.ts';

import { Model } from './def.ts';
import { PatchedAdamOptimizer } from './PatchedAdamOptimizer.ts';

export const CONTROLLER_FEATURES_DIM = 5;
export const BATTLE_FEATURES_DIM = 4;
export const TANK_FEATURES_DIM = 8;
export const ENEMY_SLOTS = MAX_ENEMIES;
export const ENEMY_FEATURES_DIM = ENEMY_BUFFER - 1; // -1 потому что id не считаем
export const ALLY_SLOTS = MAX_ALLIES;
export const ALLY_FEATURES_DIM = ALLY_BUFFER - 1; // -1 потому что id не считаем
export const BULLET_SLOTS = MAX_BULLETS;
export const BULLET_FEATURES_DIM = BULLET_BUFFER - 1; // -1 потому что id не считаем

export function createPolicyNetwork(): tf.LayersModel {
    const { inputs, network } = createBaseNetwork(Model.Policy, 32, 1);
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
    const { inputs, network } = createBaseNetwork(Model.Policy, 16, 1);
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

function createBaseNetwork(modelName: Model, dModel: number, heads: number) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, dModel);

    const tankToEnemiesAttn = applyCrossAttentionLayer(modelName + '_tankToEnemiesAttn', heads, {
        qTok: tokens.tankTok,
        kvTok: tokens.enemiesTok,
        kvMask: inputs.enemiesMaskInput,
    });
    const tankToAlliesAttn = applyCrossAttentionLayer(modelName + '_tankToAlliesAttn', heads, {
        qTok: tokens.tankTok,
        kvTok: tokens.alliesTok,
        kvMask: inputs.alliesMaskInput,
    });
    const tankToBulletsAttn = applyCrossAttentionLayer(modelName + '_tankToBulletsAttn', heads, {
        qTok: tokens.tankTok,
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
    });

    const envToken = tf.layers.concatenate({ name: modelName + '_envToken', axis: 1 }).apply([
        tokens.controllerTok,
        tokens.battleTok,
        tokens.tankTok,
        tankToEnemiesAttn,
        tankToAlliesAttn,
        tankToBulletsAttn,
    ]) as tf.SymbolicTensor;

    const selfAttn = applySelfAttentionLayer(modelName + '_envTokenSelfAttn', heads, {
        tokens: envToken,
    });

    const normSelfAttn = tf.layers.layerNormalization({ name: modelName + 'normEnvToken' }).apply(selfAttn) as tf.SymbolicTensor;
    const finalToken = tf.layers.flatten({ name: modelName + '_flattenEnvToken' }).apply(normSelfAttn) as tf.SymbolicTensor;

    const finalTokenDim = finalToken.shape[1]!;
    const withDenseLayers = applyDenseLayers(
        finalToken,
        [['relu', finalTokenDim * 2], ['relu', finalTokenDim]],
    );

    return { inputs, network: withDenseLayers };
}