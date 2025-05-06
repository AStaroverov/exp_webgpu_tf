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
import { applyDenseLayers, applySelfAttentionLayer, createInputs } from './Layers.ts';

import { Model } from './def.ts';
import { PatchedAdamOptimizer } from './PatchedAdamOptimizer.ts';

export const BATTLE_FEATURES_DIM = 4;
export const TANK_FEATURES_DIM = 8;
export const ENEMY_SLOTS = MAX_ENEMIES;
export const ENEMY_FEATURES_DIM = ENEMY_BUFFER - 1; // -1 потому что id не считаем
export const ALLY_SLOTS = MAX_ALLIES;
export const ALLY_FEATURES_DIM = ALLY_BUFFER - 1; // -1 потому что id не считаем
export const BULLET_SLOTS = MAX_BULLETS;
export const BULLET_FEATURES_DIM = BULLET_BUFFER - 1; // -1 потому что id не считаем

export function createPolicyNetwork(): tf.LayersModel {
    const dModel = 32;
    const inputs = createInputs(Model.Policy);
    const attentionLayer = applySelfAttentionLayer(Model.Policy, dModel, inputs);
    const withDenseLayers = applyDenseLayers(attentionLayer, [['relu', dModel * 4], ['relu', dModel * 2]]);
    // Выход: ACTION_DIM * 2 (пример: mean и logStd) ---
    const policyOutput = tf.layers.dense({
        name: Model.Policy + '_output',
        units: ACTION_DIM * 2,
        activation: 'linear',
    }).apply(withDenseLayers) as tf.SymbolicTensor;
    const inputsArr = Object.values(inputs);
    const model = tf.model({
        name: Model.Policy,
        inputs: inputsArr,
        outputs: policyOutput,
    });
    model.optimizer = new PatchedAdamOptimizer(CONFIG.lrConfig.initial);
    // fake loss for save optimizer with model
    model.loss = 'meanSquaredError';

    return model;
}

export function createValueNetwork(): tf.LayersModel {
    const dModel = 16;
    const inputs = createInputs(Model.Value);
    const attentionLayer = applySelfAttentionLayer(Model.Value, dModel, inputs);
    const withDenseLayers = applyDenseLayers(attentionLayer, [['relu', dModel * 4], ['relu', dModel * 2]]);
    const valueOutput = tf.layers.dense({
        name: Model.Value + '_output',
        units: 1,
        activation: 'linear',
    }).apply(withDenseLayers) as tf.SymbolicTensor;
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
