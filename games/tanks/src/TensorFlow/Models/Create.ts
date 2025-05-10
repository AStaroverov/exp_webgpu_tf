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
import { applyCrossAttentionLayer, applyDenseLayers, convertInputsToTokens, createInputs } from './Layers.ts';

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

    const tankToEnemiesAttn = applyCrossAttentionLayer(modelName + '_tankToEnemiesAttn', dModel, heads, {
        qTok: tokens.tankTok,
        kvTok: tokens.enemiesTok,
        kvMask: inputs.enemiesMaskInput,
    });
    const tankToAlliesAttn = applyCrossAttentionLayer(modelName + '_tankToAlliesAttn', dModel, heads, {
        qTok: tokens.tankTok,
        kvTok: tokens.alliesTok,
        kvMask: inputs.alliesMaskInput,
    });
    const tankToBulletsAttn = applyCrossAttentionLayer(modelName + '_tankToBulletsAttn', dModel, heads, {
        qTok: tokens.tankTok,
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
    });

    // const envToken = tf.layers.concatenate({ name: modelName + '_envToken' }).apply([
    //     tokens.tankTok,
    //     tokens.battleTok,
    //     tankToEnemiesAttn,
    //     tankToAlliesAttn,
    //     tankToBulletsAttn,
    // ]) as tf.SymbolicTensor;
    // // TODO: self-attention over envToken
    //
    // const controllerToEnvAttn = applyCrossAttentionLayer(modelName + '_controllerToEnvAttn', dModel, heads, {
    //     qTok: tokens.controllerTok,
    //     kvTok: envToken,
    // });
    //
    // const withDenseLayers = applyDenseLayers(
    //     tf.layers.flatten().apply(controllerToEnvAttn) as tf.SymbolicTensor,
    //     [['relu', dModel * 2], ['relu', dModel]],
    // );

    const envToken = tf.layers.concatenate({ name: modelName + '_envToken' }).apply([
        tokens.controllerTok,
        tokens.battleTok,
        tokens.tankTok,
        tankToEnemiesAttn,
        tankToAlliesAttn,
        tankToBulletsAttn,
    ]) as tf.SymbolicTensor;
    const normEnvToken = tf.layers.layerNormalization({ name: modelName + 'normEnvToken' }).apply(envToken) as tf.SymbolicTensor;
    const flattenedEnvToken = tf.layers.flatten({ name: modelName + '_flattenEnvToken' }).apply(normEnvToken) as tf.SymbolicTensor;

    const withDenseLayers = applyDenseLayers(
        flattenedEnvToken,
        [['relu', flattenedEnvToken.shape[1]! * 2], ['relu', flattenedEnvToken.shape[1]!]],
    );

    return { inputs, network: withDenseLayers };
}