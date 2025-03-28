import * as tf from '@tensorflow/tfjs';
import {
    TANK_INPUT_TENSOR_BULLET_BUFFER,
    TANK_INPUT_TENSOR_ENEMY_BUFFER,
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
} from '../../ECS/Components/TankState.ts';
import { ACTION_DIM } from './consts.ts';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { SymbolicTensor } from '@tensorflow/tfjs-layers/dist/engine/topology';

export const TANK_FEATURES_DIM = 7; // пример
export const ENEMY_FEATURES_DIM = TANK_INPUT_TENSOR_ENEMY_BUFFER;   // 7
export const ENEMY_SLOTS = TANK_INPUT_TENSOR_MAX_ENEMIES;           // 4 (допустим)
export const BULLET_FEATURES_DIM = TANK_INPUT_TENSOR_BULLET_BUFFER; // 5
export const BULLET_SLOTS = TANK_INPUT_TENSOR_MAX_BULLETS;          // 4 (допустим)

export function createPolicyNetwork(
    hiddenLayers: [ActivationIdentifier, number][],
): tf.LayersModel {
    const { inputs, merged } = createInputLayer();

    let x = merged;
    let i = 0;
    for (const [activation, units] of hiddenLayers) {
        x = tf.layers.dense({
            name: `policy/dense${ i++ }`,
            units,
            activation,
            kernelInitializer: 'glorotUniform',
        }).apply(x) as tf.SymbolicTensor;
    }

    // Выход: ACTION_DIM * 2 (пример: mean и logStd) ---
    const policyOutput = tf.layers.dense({
        name: 'policy/output',
        units: ACTION_DIM * 2,
        activation: 'linear',
    }).apply(x) as tf.SymbolicTensor;

    return tf.model({
        name: 'policy',
        inputs: inputs,
        outputs: policyOutput,
    });
}

// Аналогично для Value-сети
export function createValueNetwork(
    hiddenLayers: [ActivationIdentifier, number][],
): tf.LayersModel {
    const { inputs, merged } = createInputLayer();

    let x = merged;
    let i = 0;
    for (const [activation, units] of hiddenLayers) {
        x = tf.layers.dense({
            name: `value/dense${ i++ }`,
            units,
            activation,
            kernelInitializer: 'glorotUniform',
        }).apply(x) as tf.SymbolicTensor;
    }

    const valueOutput = tf.layers.dense({
        name: 'value/output',
        units: 1,
        activation: 'linear',
    }).apply(x) as tf.SymbolicTensor;

    return tf.model({
        name: 'value',
        inputs: inputs,
        outputs: valueOutput,
    });
}

function createInputLayer() {
    // --- 1) Создаём входы ---
    const tankInput = tf.input({
        name: 'tankInput',
        shape: [TANK_FEATURES_DIM],
    });
    // Вход для врагов: 2D [ENEMY_SLOTS, ENEMY_FEATURES_DIM]
    const enemiesInput = (tf.input({
        name: 'enemiesInput',
        shape: [ENEMY_SLOTS, ENEMY_FEATURES_DIM],
    }));
    // Вход для пуль: 2D [BULLET_SLOTS, BULLET_FEATURES_DIM]
    const bulletsInput = (tf.input({
        name: 'bulletsInput',
        shape: [BULLET_SLOTS, BULLET_FEATURES_DIM],
    }));

    // --- 3) Склеиваем все три вектора вместе [tank + enemies + bullets] ---
    return {
        inputs: [tankInput, enemiesInput, bulletsInput],
        merged: tf.layers
            .concatenate()
            .apply([tankInput, asDynamic(enemiesInput), asDynamic(bulletsInput)]) as tf.SymbolicTensor,
    };
}

function asDynamic(layer: SymbolicTensor): tf.SymbolicTensor {
    const masked = tf.layers.masking({ maskValue: 0 }).apply(layer);
    const lstm = tf.layers.lstm({ units: 16, returnSequences: false }).apply(masked);

    return lstm as tf.SymbolicTensor;
}