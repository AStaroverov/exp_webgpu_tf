import * as tf from '@tensorflow/tfjs';
import { Tensor } from '@tensorflow/tfjs';
import {
    TANK_INPUT_TENSOR_BULLET_BUFFER,
    TANK_INPUT_TENSOR_ENEMY_BUFFER,
    TANK_INPUT_TENSOR_MAX_BULLETS,
    TANK_INPUT_TENSOR_MAX_ENEMIES,
} from '../../ECS/Components/TankState.ts';
import { ACTION_DIM } from './consts.ts';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

export const TANK_FEATURES_DIM = 7;
export const ENEMY_SLOTS = TANK_INPUT_TENSOR_MAX_ENEMIES;
export const ENEMY_FEATURES_DIM = TANK_INPUT_TENSOR_ENEMY_BUFFER - 1; // -1 потому что id не считаем
export const BULLET_SLOTS = TANK_INPUT_TENSOR_MAX_BULLETS;
export const BULLET_FEATURES_DIM = TANK_INPUT_TENSOR_BULLET_BUFFER;

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
    const tankInput = tf.input({ name: 'tankInput', shape: [TANK_FEATURES_DIM] });
    const enemiesMask = tf.input({ name: 'enemyMaskInput', shape: [ENEMY_SLOTS] });
    const enemiesInput = tf.input({ name: 'enemiesInput', shape: [ENEMY_SLOTS, ENEMY_FEATURES_DIM] });
    const bulletsInput = tf.input({ name: 'bulletsInput', shape: [BULLET_SLOTS, BULLET_FEATURES_DIM] });

    const enemiesAttentionContext = applyEnemyAttention(tankInput, enemiesInput, enemiesMask);

    const bulletsFlat = tf.layers.flatten().apply(bulletsInput) as tf.SymbolicTensor;

    const merged = tf.layers.concatenate().apply([tankInput, enemiesAttentionContext, bulletsFlat]) as tf.SymbolicTensor;

    return {
        inputs: [tankInput, enemiesMask, enemiesInput, bulletsInput],
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

function applyEnemyAttention(tankInput: tf.SymbolicTensor, enemiesInput: tf.SymbolicTensor, enemiesMaskInput: tf.SymbolicTensor, dModel = 32): tf.SymbolicTensor {
    const denseQ = tf.layers.dense({ units: dModel, useBias: false });
    const denseK = tf.layers.dense({ units: dModel, useBias: false });
    const denseV = tf.layers.dense({ units: dModel, useBias: false });

    const Q = denseQ.apply(tankInput) as tf.SymbolicTensor;                        // [batch, d_model]
    const K = denseK.apply(enemiesInput) as tf.SymbolicTensor;                    // [batch, ENEMY_SLOTS, d_model]
    const V = denseV.apply(enemiesInput) as tf.SymbolicTensor;

    const Q_reshaped = tf.layers.reshape({ targetShape: [1, dModel] }).apply(Q) as tf.SymbolicTensor; // [batch, 1, d_model]

    const scores = tf.layers.dot({ axes: -1 }).apply([Q_reshaped, K]) as tf.SymbolicTensor;

    const maskedScores = new AttentionMaskLayer().apply([scores, enemiesMaskInput]) as tf.SymbolicTensor;

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
}

tf.serialization.registerClass(AttentionMaskLayer);