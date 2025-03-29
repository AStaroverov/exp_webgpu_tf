import * as tf from '@tensorflow/tfjs';
import {
    TANK_INPUT_TENSOR_BULLET_BUFFER,
    TANK_INPUT_TENSOR_ENEMY_BUFFER,
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
} from '../../ECS/Components/TankState.ts';
import { ACTION_DIM } from './consts.ts';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';

export const TANK_FEATURES_DIM = 7;
export const ENEMY_FEATURES_DIM = TANK_INPUT_TENSOR_ENEMY_BUFFER;
export const ENEMY_SLOTS = TANK_INPUT_TENSOR_MAX_ENEMIES;
export const BULLET_FEATURES_DIM = TANK_INPUT_TENSOR_BULLET_BUFFER;
export const BULLET_SLOTS = TANK_INPUT_TENSOR_MAX_BULLETS;

const denseLayerPolicyEnemies: [ActivationIdentifier, number][] = [
    ['relu', 64],// 2 * ENEMY_SLOTS * ENEMY_FEATURES_DIM],
    ['relu', 32],//1 * ENEMY_SLOTS * ENEMY_FEATURES_DIM],
];
const denseLayerPolicyBullets: [ActivationIdentifier, number][] = [
    ['relu', 64],//2 * BULLET_SLOTS * BULLET_FEATURES_DIM],
    ['relu', 32],//1 * BULLET_SLOTS * BULLET_FEATURES_DIM],
];
const denseLayersPolicy: [ActivationIdentifier, number][] = [['relu', 128], ['relu', 64], ['relu', 32]];
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

    return tf.model({
        name: 'policy',
        inputs: inputs,
        outputs: policyOutput,
    });
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
    const enemiesInput =
        tf.input({
            name: 'enemiesInput',
            shape: [ENEMY_SLOTS, ENEMY_FEATURES_DIM],
        });
    const enemiesLayers = addDenseLayers(
        tf.layers.flatten().apply(enemiesInput) as tf.SymbolicTensor,
        denseLayerPolicyEnemies,
    );
    // Вход для пуль: 2D [BULLET_SLOTS, BULLET_FEATURES_DIM]
    const bulletsInput =
        tf.input({
            name: 'bulletsInput',
            shape: [BULLET_SLOTS, BULLET_FEATURES_DIM],
        });
    const bulletsLayers = addDenseLayers(
        tf.layers.flatten().apply(bulletsInput) as tf.SymbolicTensor,
        denseLayerPolicyBullets,
    );
    // --- 2) Склеиваем все три вектора вместе [tank + enemies + bullets] ---
    return {
        inputs: [tankInput, enemiesInput, bulletsInput],
        merged: tf.layers.concatenate().apply([tankInput, enemiesLayers, bulletsLayers]) as tf.SymbolicTensor,
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