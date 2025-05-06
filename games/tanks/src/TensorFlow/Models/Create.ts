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
import { applyDenseLayers } from './Layers.ts';

import { Model } from './def.ts';
import { MultiHeadSelfAttentionLayer } from './MultiHeadSelfAttentionLayer.ts';
import { SymbolicTensor } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { OnesMask } from './ConstOnesMaskLayer.ts';

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
    const attentionLayer = createAttentionLayer(Model.Policy, 32, inputs);
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
    const attentionLayer = createAttentionLayer(Model.Value, 16, inputs);
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

function createInputs(name: string) {
    const battleInput = tf.input({ name: name + '_battlefieldInput', shape: [BATTLE_FEATURES_DIM] });
    const tankInput = tf.input({ name: name + '_tankInput', shape: [TANK_FEATURES_DIM] });

    const enemiesInput = tf.input({ name: name + '_enemiesInput', shape: [ENEMY_SLOTS, ENEMY_FEATURES_DIM] });
    const enemiesMaskInput = tf.input({ name: name + '_enemiesMaskInput', shape: [ENEMY_SLOTS] });
    const alliesInput = tf.input({ name: name + '_alliesInput', shape: [ALLY_SLOTS, ALLY_FEATURES_DIM] });
    const alliesMaskInput = tf.input({ name: name + '_alliesMaskInput', shape: [ALLY_SLOTS] });
    const bulletsInput = tf.input({ name: name + '_bulletsInput', shape: [BULLET_SLOTS, BULLET_FEATURES_DIM] });
    const bulletsMaskInput = tf.input({ name: name + '_bulletsMaskInput', shape: [BULLET_SLOTS] });

    return {
        battleInput,
        tankInput,
        enemiesInput,
        enemiesMaskInput,
        alliesInput,
        alliesMaskInput,
        bulletsInput,
        bulletsMaskInput,
    };
}

function proj(x: tf.SymbolicTensor, dModel: number, name: string) {
    return tf.layers.dense({ units: dModel, activation: 'relu', name: name + '_token' }).apply(x);
}

function createAttentionLayer(
    name: string,
    dModel: number, // 32 | 64 |....
    {
        battleInput,
        tankInput,
        alliesInput,
        enemiesInput,
        bulletsInput,
        alliesMaskInput,
        enemiesMaskInput,
        bulletsMaskInput,
    }: ReturnType<typeof createInputs>,
) {
    name = name + '_AttentionLayer';

    // ---------- embed everything to a common d_model ---------------------------
    const battleTok = tf.layers.reshape({ targetShape: [1, dModel] })
        .apply(proj(battleInput, dModel, name + '_' + battleInput.name));
    const tankTok = tf.layers.reshape({ targetShape: [1, dModel] })
        .apply(proj(tankInput, dModel, name + '_' + tankInput.name));
    const allyTok = proj(alliesInput, dModel, name + '_' + alliesInput.name);   // [B,3,d]
    const enemyTok = proj(enemiesInput, dModel, name + '_' + enemiesInput.name);  // [B,4,d]
    const bulletTok = proj(bulletsInput, dModel, name + '_' + bulletsInput.name);  // [B,N,d]

    const tokens = tf.layers.concatenate({ name: name + '_tokens', axis: 1 })
        .apply([battleTok, tankTok, allyTok, enemyTok, bulletTok] as tf.SymbolicTensor[]) as tf.SymbolicTensor;            // [B,S,d]

    // ---------- build 0/1 padding mask -----------------------------------------
    const battleInputFixedMask = new OnesMask({ name: name + '_battleInputFixedMask' }).apply(battleInput) as tf.SymbolicTensor;   // [dModel,2]
    const tankInputFixedMask = new OnesMask({ name: name + '_tankInputFixedMask' }).apply(tankInput) as tf.SymbolicTensor;   // [dModel,2]
    const mask = tf.layers.concatenate({ name: name + '_masks', axis: 1 })
        .apply([
            battleInputFixedMask,
            tankInputFixedMask,
            alliesMaskInput,
            enemiesMaskInput,
            bulletsMaskInput,
        ] as SymbolicTensor[]) as tf.SymbolicTensor;           // [B,S]

    // ---------- self-attention block -------------------------------------------
    if (dModel % 16 !== 0) {
        throw new Error('Dim model for attention layer must be pow of 2 and more than 16')
    }
    let x = new MultiHeadSelfAttentionLayer({
        name: name + '_MultiHeadSelfAttentionLayer',
        numHeads: dModel / 16,
        keyDim: 16,
    }).apply([tokens, mask]) as tf.SymbolicTensor;

    x = tf.layers.add({ name: name + '_add' }).apply([x, tokens]) as tf.SymbolicTensor;
    x = tf.layers.layerNormalization({ name: name + 'normalization', epsilon: 1e-5 }).apply(x) as tf.SymbolicTensor;

    return tf.layers.flatten({ name: name + '_flatten' }).apply(x) as tf.SymbolicTensor;
}
