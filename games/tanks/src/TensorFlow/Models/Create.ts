import * as tf from '@tensorflow/tfjs';
import {
    ALLY_BUFFER,
    BULLET_BUFFER,
    ENEMY_BUFFER,
    MAX_ALLIES,
    MAX_BULLETS,
    MAX_ENEMIES,
} from '../../ECS/Components/Tank/TankState.ts';
import { ACTION_DIM } from '../Common/consts.ts';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { CONFIG } from '../PPO/config.ts';
import { applyAttentionLayer, applyDenseLayers } from './Layers.ts';
import { Model } from './Transfer.ts';

export const BATTLE_FEATURES_DIM = 4;
export const TANK_FEATURES_DIM = 7;
export const ENEMY_SLOTS = MAX_ENEMIES;
export const ENEMY_FEATURES_DIM = ENEMY_BUFFER - 1; // -1 потому что id не считаем
export const ALLY_SLOTS = MAX_ALLIES;
export const ALLY_FEATURES_DIM = ALLY_BUFFER - 1; // -1 потому что id не считаем
export const BULLET_SLOTS = MAX_BULLETS;
export const BULLET_FEATURES_DIM = BULLET_BUFFER - 1; // -1 потому что id не считаем

const denseLayersPolicy: [ActivationIdentifier, number][] = [['relu', 256], ['relu', 128], ['relu', 64]];
const denseLayersValue: [ActivationIdentifier, number][] = [['relu', 64], ['relu', 32]];

export function createPolicyNetwork(): tf.LayersModel {
    const { inputs, merged } = createInputLayer();
    const withDenseLayers = applyDenseLayers(merged, denseLayersPolicy);
    // Выход: ACTION_DIM * 2 (пример: mean и logStd) ---
    const policyOutput = tf.layers.dense({
        name: 'policy/output',
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

// Аналогично для Value-сети
export function createValueNetwork(): tf.LayersModel {
    const { inputs, merged } = createInputLayer();
    const withDenseLayers = applyDenseLayers(merged, denseLayersValue);
    const valueOutput = tf.layers.dense({
        name: 'value/output',
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

function createInputLayer() {
    const battleInput = tf.input({ name: 'battlefieldInput', shape: [BATTLE_FEATURES_DIM] });
    const tankInput = tf.input({ name: 'tankInput', shape: [TANK_FEATURES_DIM] });

    const enemiesInput = tf.input({ name: 'enemiesInput', shape: [ENEMY_SLOTS, ENEMY_FEATURES_DIM] });
    const enemiesMaskInput = tf.input({ name: 'enemiesMaskInput', shape: [ENEMY_SLOTS] });
    const alliesInput = tf.input({ name: 'alliesInput', shape: [ALLY_SLOTS, ALLY_FEATURES_DIM] });
    const alliesMaskInput = tf.input({ name: 'alliesMaskInput', shape: [ALLY_SLOTS] });
    const bulletsInput = tf.input({ name: 'bulletsInput', shape: [BULLET_SLOTS, BULLET_FEATURES_DIM] });
    const bulletsMaskInput = tf.input({ name: 'bulletsMaskInput', shape: [BULLET_SLOTS] });

    const enemiesAttentionContext = applyAttentionLayer('enemies', tankInput, enemiesInput, enemiesMaskInput);
    const alliesAttentionContext = applyAttentionLayer('allies', tankInput, alliesInput, alliesMaskInput);
    const bulletsAttentionContext = applyAttentionLayer('bullets', tankInput, bulletsInput, bulletsMaskInput);

    const merged = tf.layers.concatenate().apply([
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

