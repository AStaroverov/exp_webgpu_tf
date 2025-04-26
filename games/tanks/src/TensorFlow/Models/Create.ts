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
import { applyAttentionLayer, applyDenseLayers } from './Layers.ts';
import { Model } from './Transfer.ts';

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
    const { inputs, merged } = createInputLayer(Model.Policy);
    const withDenseLayers = applyDenseLayers(merged, denseLayersPolicy);
    // Выход: ACTION_DIM * 2 (пример: mean и logStd) ---
    const policyOutput = tf.layers.dense({
        name: Model.Policy + '_output',
        units: ACTION_DIM * 2,
        activation: 'linear',
    }).apply(withDenseLayers) as tf.SymbolicTensor;
    const model = tf.model({
        name: Model.Policy,
        inputs: inputs,
        outputs: policyOutput,
    });
    model.optimizer = tf.train.adam(CONFIG.lrConfig.initial);
    // fake loss for save optimizer with model
    model.loss = 'meanSquaredError';

    return model;
}

export function createValueNetwork(): tf.LayersModel {
    const { inputs, merged } = createInputLayer(Model.Value);
    const withDenseLayers = applyDenseLayers(merged, denseLayersValue);
    const valueOutput = tf.layers.dense({
        name: Model.Value + '_output',
        units: 1,
        activation: 'linear',
    }).apply(withDenseLayers) as tf.SymbolicTensor;
    const model = tf.model({
        name: Model.Value,
        inputs: inputs,
        outputs: valueOutput,
    });
    model.optimizer = tf.train.adam(CONFIG.lrConfig.initial);
    // fake loss for save optimizer with model
    model.loss = 'meanSquaredError';

    return model;
}

function createInputLayer(name: string) {
    const battleInput = tf.input({ name: name + '_battlefieldInput', shape: [BATTLE_FEATURES_DIM] });
    const tankInput = tf.input({ name: name + '_tankInput', shape: [TANK_FEATURES_DIM] });

    const enemiesInput = tf.input({ name: name + '_enemiesInput', shape: [ENEMY_SLOTS, ENEMY_FEATURES_DIM] });
    const enemiesMaskInput = tf.input({ name: name + '_enemiesMaskInput', shape: [ENEMY_SLOTS] });
    const alliesInput = tf.input({ name: name + '_alliesInput', shape: [ALLY_SLOTS, ALLY_FEATURES_DIM] });
    const alliesMaskInput = tf.input({ name: name + '_alliesMaskInput', shape: [ALLY_SLOTS] });
    const bulletsInput = tf.input({ name: name + '_bulletsInput', shape: [BULLET_SLOTS, BULLET_FEATURES_DIM] });
    const bulletsMaskInput = tf.input({ name: name + '_bulletsMaskInput', shape: [BULLET_SLOTS] });

    const enemiesAttentionContext = applyAttentionLayer(name + '_enemies', tankInput, enemiesInput, enemiesMaskInput);
    const alliesAttentionContext = applyAttentionLayer(name + '_allies', tankInput, alliesInput, alliesMaskInput);
    const bulletsAttentionContext = applyAttentionLayer(name + '_bullets', tankInput, bulletsInput, bulletsMaskInput);

    const merged = tf.layers.concatenate({
        name: name + '_merged',
    }).apply([
        battleInput,
        tankInput,
        alliesAttentionContext,
        enemiesAttentionContext,
        bulletsAttentionContext,
    ]) as tf.SymbolicTensor;

    return {
        inputs: [
            battleInput,
            tankInput,
            enemiesInput,
            enemiesMaskInput,
            alliesInput,
            alliesMaskInput,
            bulletsInput,
            bulletsMaskInput,
        ],
        merged,
    };
}
