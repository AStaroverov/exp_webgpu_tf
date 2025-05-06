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
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { CONFIG } from '../PPO/config.ts';
import { applyDenseLayers, applySelfAttentionLayer, createInputs } from './Layers.ts';

import { Model } from './def.ts';

export const BATTLE_FEATURES_DIM = 4;
export const TANK_FEATURES_DIM = 8;
export const ENEMY_SLOTS = MAX_ENEMIES;
export const ENEMY_FEATURES_DIM = ENEMY_BUFFER - 1; // -1 потому что id не считаем
export const ALLY_SLOTS = MAX_ALLIES;
export const ALLY_FEATURES_DIM = ALLY_BUFFER - 1; // -1 потому что id не считаем
export const BULLET_SLOTS = MAX_BULLETS;
export const BULLET_FEATURES_DIM = BULLET_BUFFER - 1; // -1 потому что id не считаем

const denseLayersPolicy: [ActivationIdentifier, number][] = [['relu', 256], ['relu', 128], ['relu', 64]];
const denseLayersValue: [ActivationIdentifier, number][] = [['relu', 64], ['relu', 32]];

export function createPolicyNetwork(): tf.LayersModel {
    const inputs = createInputs(Model.Policy);
    const attentionLayer = applySelfAttentionLayer(Model.Policy, 32, inputs);
    const withDenseLayers = applyDenseLayers(attentionLayer, denseLayersPolicy);
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
    model.optimizer = tf.train.adam(CONFIG.lrConfig.initial);
    // fake loss for save optimizer with model
    model.loss = 'meanSquaredError';

    return model;
}

export function createValueNetwork(): tf.LayersModel {
    const inputs = createInputs(Model.Value);
    const attentionLayer = applySelfAttentionLayer(Model.Value, 16, inputs);
    const withDenseLayers = applyDenseLayers(attentionLayer, denseLayersValue);
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
    model.optimizer = tf.train.adam(CONFIG.lrConfig.initial);
    // fake loss for save optimizer with model
    model.loss = 'meanSquaredError';

    return model;
}
