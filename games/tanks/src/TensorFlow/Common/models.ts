import * as tf from '@tensorflow/tfjs';
import { Tensor } from '@tensorflow/tfjs';
import {
    ALLY_BUFFER,
    BULLET_BUFFER,
    ENEMY_BUFFER,
    MAX_ALLIES,
    MAX_BULLETS,
    MAX_ENEMIES,
} from '../../ECS/Components/TankState.ts';
import { ACTION_DIM } from './consts.ts';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { CONFIG } from '../PPO/Common/config.ts';

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
    const withDenseLayers = addDenseLayers(merged, denseLayersPolicy);
    // Выход: ACTION_DIM * 2 (пример: mean и logStd) ---
    const policyOutput = tf.layers.dense({
        name: 'policy/output',
        units: ACTION_DIM * 2,
        activation: 'linear',
    }).apply(withDenseLayers) as tf.SymbolicTensor;
    const model = tf.model({
        name: 'policy',
        inputs: inputs,
        outputs: policyOutput,
    });
    model.optimizer = tf.train.adam(CONFIG.lrConfig.initial);

    return model;
}

// Аналогично для Value-сети
export function createValueNetwork(): tf.LayersModel {
    const { inputs, merged } = createInputLayer();
    const withDenseLayers = addDenseLayers(merged, denseLayersValue);
    const valueOutput = tf.layers.dense({
        name: 'value/output',
        units: 1,
        activation: 'linear',
    }).apply(withDenseLayers) as tf.SymbolicTensor;
    const model = tf.model({
        name: 'value',
        inputs: inputs,
        outputs: valueOutput,
    });
    model.optimizer = tf.train.adam(CONFIG.lrConfig.initial);

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

    const enemiesAttentionContext = applyAttention('enemies', tankInput, enemiesInput, enemiesMaskInput);
    const alliesAttentionContext = applyAttention('allies', tankInput, alliesInput, alliesMaskInput);
    const bulletsAttentionContext = applyAttention('bullets', tankInput, bulletsInput, bulletsMaskInput);

    const merged = tf.layers.concatenate().apply([
        battleInput,
        tankInput,
        alliesAttentionContext,
        enemiesAttentionContext,
        bulletsAttentionContext,
    ]) as tf.SymbolicTensor;

    return {
        inputs: [battleInput, tankInput, enemiesInput, enemiesMaskInput, alliesInput, alliesMaskInput, bulletsInput, bulletsMaskInput],
        merged,
    };
}

function addDenseLayers(layer: tf.SymbolicTensor, hiddenLayers: [ActivationIdentifier, number][]) {
    let x = layer;
    let i = 0;
    for (const [activation, units] of hiddenLayers) {
        x = tf.layers.dense({
            name: `${ layer.name }/dense${ i++ }`,
            units,
            activation,
            kernelInitializer: 'glorotUniform',
        }).apply(x) as tf.SymbolicTensor;
    }

    return x;
}

function applyAttention(name: string, qInput: tf.SymbolicTensor, kvInput: tf.SymbolicTensor, kvMaskInput: tf.SymbolicTensor, dModel = 32): tf.SymbolicTensor {
    const denseQ = tf.layers.dense({ name: name + '_denseQ', units: dModel, useBias: false });
    const denseK = tf.layers.dense({ name: name + '_denseK', units: dModel, useBias: false });
    const denseV = tf.layers.dense({ name: name + '_denseV', units: dModel, useBias: false });

    const Q = denseQ.apply(qInput) as tf.SymbolicTensor;  // [batch, dim model]
    const K = denseK.apply(kvInput) as tf.SymbolicTensor;  // [batch, slots, dim model]
    const V = denseV.apply(kvInput) as tf.SymbolicTensor;

    const Q_reshaped = tf.layers.reshape({ targetShape: [1, dModel] }).apply(Q) as tf.SymbolicTensor; // [batch, 1, d_model]

    const scores = tf.layers.dot({ axes: -1 }).apply([Q_reshaped, K]) as tf.SymbolicTensor;

    const maskedScores = new AttentionMaskLayer().apply([scores, kvMaskInput]) as tf.SymbolicTensor;

    const attnWeights = tf.layers.activation({ activation: 'softmax' }).apply(maskedScores) as tf.SymbolicTensor;

    const context = tf.layers.dot({ axes: [2, 1] }).apply([attnWeights, V]) as tf.SymbolicTensor;

    return tf.layers.flatten().apply(context) as tf.SymbolicTensor; // [batch, d_model]
}

class AttentionMaskLayer extends tf.layers.Layer {
    constructor(config?: LayerArgs) {
        super(config);
    }

    static get className() {
        return 'AttentionMaskLayer';
    }

    call(inputs: Tensor[] | Tensor): Tensor {
        const [scores, mask] = inputs as Tensor[]; // scores: [batch, 1, slots], mask: [batch, slots]

        return tf.tidy(() => {
            const one = tf.scalar(1);
            const negInf = tf.scalar(-1e9);

            if (mask.shape.length !== 2) {
                throw new Error('mask shape must be 2D');
            }
            const maskExpanded = mask.reshape([-1, 1, mask.shape[1]]); // [batch, 1, slots]
            const inverted = tf.sub(one, maskExpanded);                // 1 - mask
            const penalty = tf.mul(inverted, negInf);                  // (1 - mask) * -1e9

            return tf.add(scores, penalty); // scores + penalty
        });
    }

    computeOutputShape(inputShape: [number[], number[]]): number[] {
        return inputShape[0]; // output shape = scores shape
    }

    getConfig() {
        return { ...super.getConfig() };
    }
}

tf.serialization.registerClass(AttentionMaskLayer);
