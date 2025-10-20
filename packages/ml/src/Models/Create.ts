import * as tf from '@tensorflow/tfjs';
import { CONFIG } from '../../../ml-common/config.ts';
import { ACTION_DIM } from '../../../ml-common/consts.ts';
import {
    ALLY_BUFFER,
    BULLET_BUFFER,
    ENEMY_BUFFER,
    MAX_ALLIES,
    MAX_BULLETS,
    MAX_ENEMIES,
} from '../../../tanks/src/Pilots/Components/TankState.ts';
import {
    applyCrossTransformerLayer,
    applyDenseLayers,
    applySelfTransformLayers,
    convertInputsToTokens,
    createInputs,
} from './ApplyLayers.ts';

import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { Model } from './def.ts';
import { PatchedAdamOptimizer } from './PatchedAdamOptimizer.ts';

export const CONTROLLER_FEATURES_DIM = 4;
export const BATTLE_FEATURES_DIM = 6;
export const TANK_FEATURES_DIM = 7;
export const ENEMY_SLOTS = MAX_ENEMIES;
export const ENEMY_FEATURES_DIM = ENEMY_BUFFER - 1; // -1 потому что id не считаем
export const ALLY_SLOTS = MAX_ALLIES;
export const ALLY_FEATURES_DIM = ALLY_BUFFER - 1; // -1 потому что id не считаем
export const BULLET_SLOTS = MAX_BULLETS;
export const BULLET_FEATURES_DIM = BULLET_BUFFER - 1; // -1 потому что id не считаем

type NetworkConfig = {
    dim: number;
    heads: number;
    dropout?: number;
    finalMLP: [ActivationIdentifier, number][];
};

type policyNetworkConfig = NetworkConfig & {
    headMLP: {
        mean: [ActivationIdentifier, number][];
        logStd: [ActivationIdentifier, number][];
    };
};

const policyNetworkConfig: policyNetworkConfig = {
    dim: 64,
    heads: 4,
    finalMLP: [
        ['relu', 512],
        ['relu', 256],
    ],
    headMLP: {
        mean: [
            ['relu', 256],
            ['relu', 128],
        ],
        logStd: [
            ['relu', 64],
        ],
    },
};
const valueNetworkConfig: NetworkConfig = {
    dim: 16,
    heads: 1,
    finalMLP: [
        ['relu', 128],
        ['relu', 64],
        ['relu', 32],
    ] as [ActivationIdentifier, number][],
};

// SAC Critic Network configuration
const criticNetworkConfig: NetworkConfig = {
    dim: 64,               // embedding dimension
    heads: 4,              // number of attention heads
    dropout: 0.0,          // dropout rate
    finalMLP: [
        ['relu', 256],
        ['relu', 256],
        ['relu', 128],
    ] as [ActivationIdentifier, number][],
};

export function createPolicyNetwork(): tf.LayersModel {
    const { inputs, network } = createBaseNetwork(Model.Policy, policyNetworkConfig);

    // Две головы: одна для действий (mean), другая для log_std
    // Mean head с отдельным MLP
    const meanMLP = applyDenseLayers(
        Model.Policy + '_mean_mlp',
        network,
        policyNetworkConfig.headMLP.mean,
    );
    const actionHead = tf.layers.dense({
        name: Model.Policy + '_action_head',
        units: ACTION_DIM,
        activation: 'linear',
    }).apply(meanMLP) as tf.SymbolicTensor;

    // LogStd head с отдельным MLP
    const logStdMLP = applyDenseLayers(
        Model.Policy + '_log_std_mlp',
        network,
        policyNetworkConfig.headMLP.logStd,
    );
    const logStdHead = tf.layers.dense({
        name: Model.Policy + '_log_std_head',
        units: ACTION_DIM,
        activation: 'linear',
    }).apply(logStdMLP) as tf.SymbolicTensor;

    const model = tf.model({
        name: Model.Policy,
        inputs: Object.values(inputs),
        outputs: [actionHead, logStdHead],
    });
    model.optimizer = new PatchedAdamOptimizer(CONFIG.lrConfig.initial);
    // fake loss for save optimizer with model
    model.loss = 'meanSquaredError';

    return model;
}

export function createValueNetwork(): tf.LayersModel {
    const { inputs, network } = createBaseNetwork(Model.Value, valueNetworkConfig);
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

function createBaseNetwork(modelName: Model, config: NetworkConfig) {
    const inputs = createInputs(modelName);
    const tokens = convertInputsToTokens(inputs, config.dim);

    const tankToEnemiesAttn = applyCrossTransformerLayer(modelName + '_tankToEnemiesAttn', {
        numHeads: config.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.enemiesTok,
        kvMask: inputs.enemiesMaskInput,
    });
    const tankToAlliesAttn = applyCrossTransformerLayer(modelName + '_tankToAlliesAttn', {
        numHeads: config.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.alliesTok,
        kvMask: inputs.alliesMaskInput,
    });
    const tankToBulletsAttn = applyCrossTransformerLayer(modelName + '_tankToBulletsAttn', {
        numHeads: config.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
    });

    const envToken = tf.layers.concatenate({ name: modelName + '_envToken', axis: 1 }).apply([
        tokens.controllerTok,
        tokens.battleTok,
        tokens.tankTok,
        tankToEnemiesAttn,
        tankToAlliesAttn,
        tankToBulletsAttn,
    ]) as tf.SymbolicTensor;

    const transformedEnvToken = applySelfTransformLayers(
        modelName + '_transformedEnvToken',
        {
            depth: 2,
            numHeads: config.heads,
            dropout: config.dropout,
            token: envToken,
        },
    );

    const flattenFinalToken = tf.layers.flatten({ name: modelName + '_flattenFinalToken' }).apply(transformedEnvToken) as tf.SymbolicTensor;
    const normFinalToken = tf.layers.layerNormalization({ name: modelName + '_normFinalToken' }).apply(flattenFinalToken) as tf.SymbolicTensor;

    const finalMLP = applyDenseLayers(
        modelName + '_finalMLP',
        normFinalToken,
        config.finalMLP,
    );

    return { inputs, network: finalMLP };
}

// SAC Critic Network
// Takes state + action as input, outputs Q-value
export function createCriticNetwork(modelName: Model.Critic1 | Model.Critic2 | Model.TargetCritic1 | Model.TargetCritic2): tf.LayersModel {
    // Create state inputs (same as policy network)
    const inputs = createInputs(modelName);

    // Create action input
    const actionInput = tf.input({
        shape: [ACTION_DIM],
        name: modelName + '_action_input',
        dtype: 'float32',
    });

    // Encode states through transformer
    const tokens = convertInputsToTokens(inputs, criticNetworkConfig.dim);

    const tankToEnemiesAttn = applyCrossTransformerLayer(modelName + '_tankToEnemiesAttn', {
        numHeads: criticNetworkConfig.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.enemiesTok,
        kvMask: inputs.enemiesMaskInput,
    });
    const tankToAlliesAttn = applyCrossTransformerLayer(modelName + '_tankToAlliesAttn', {
        numHeads: criticNetworkConfig.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.alliesTok,
        kvMask: inputs.alliesMaskInput,
    });
    const tankToBulletsAttn = applyCrossTransformerLayer(modelName + '_tankToBulletsAttn', {
        numHeads: criticNetworkConfig.heads,
        qTok: tokens.tankTok,
        kvTok: tokens.bulletsTok,
        kvMask: inputs.bulletsMaskInput,
    });

    const envToken = tf.layers.concatenate({ name: modelName + '_envToken', axis: 1 }).apply([
        tokens.controllerTok,
        tokens.battleTok,
        tokens.tankTok,
        tankToEnemiesAttn,
        tankToAlliesAttn,
        tankToBulletsAttn,
    ]) as tf.SymbolicTensor;

    const transformedEnvToken = applySelfTransformLayers(
        modelName + '_transformedEnvToken',
        {
            depth: 2,
            numHeads: criticNetworkConfig.heads,
            dropout: criticNetworkConfig.dropout,
            token: envToken,
        },
    );

    // Global pooling for state representation
    const stateEncoding = tf.layers.flatten({
        name: modelName + '_flatten'
    }).apply(transformedEnvToken) as tf.SymbolicTensor;

    const normStateEncoding = tf.layers.layerNormalization({
        name: modelName + '_norm'
    }).apply(stateEncoding) as tf.SymbolicTensor;

    // Concatenate state encoding + action
    const combined = tf.layers.concatenate({
        name: modelName + '_concat_state_action',
    }).apply([normStateEncoding, actionInput]) as tf.SymbolicTensor;

    // MLP layers
    const hidden = applyDenseLayers(
        modelName + '_mlp',
        combined,
        criticNetworkConfig.finalMLP,
    );

    // Q-value output (single scalar)
    const qValue = tf.layers.dense({
        name: modelName + '_q_value',
        units: 1,
        activation: 'linear',
    }).apply(hidden) as tf.SymbolicTensor;

    const model = tf.model({
        inputs: [...Object.values(inputs), actionInput],
        outputs: qValue,
        name: modelName,
    });

    model.optimizer = new PatchedAdamOptimizer(CONFIG.lrConfig.initial);
    // fake loss for save optimizer with model
    model.loss = 'meanSquaredError';

    return model;
}
