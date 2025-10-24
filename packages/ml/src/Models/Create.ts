import * as tf from '@tensorflow/tfjs';
import { CONFIG } from '../../../ml-common/config.ts';
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
    applyMLP,
    applySelfTransformLayers,
    convertInputsToTokens,
    createDenseLayer,
    createInputs,
} from './ApplyLayers.ts';

import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { ACTION_DIM } from '../../../ml-common/consts.ts';
import { Model } from './def.ts';
import { StopGradientLayer } from './Layers/StopGradientLayer.ts';
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
    finalMLP: [ActivationIdentifier, number][];
};

type policyNetworkConfig = NetworkConfig & {
    latentDim?: number;  // для gSDE features
    useGSDE?: boolean;
}

const policyNetworkConfig: policyNetworkConfig = {
    dim: 64,
    heads: 4,
    finalMLP: [
        ['relu', 512],
        ['relu', 256],
        ['relu', 128],
        ['relu', 64],
    ],
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

export function createPolicyNetwork(): tf.LayersModel {
    const { inputs, network, phi } = createBaseNetwork(Model.Policy, policyNetworkConfig);

    const meanOutput = createDenseLayer({
        name: Model.Policy + '_mean_output',
        units: ACTION_DIM,
        useBias: true,
        activation: 'linear',
    }).apply(network) as tf.SymbolicTensor;

    const phiDetached = new StopGradientLayer({
        name: Model.Policy + '_phi_stop_gradient'
    }).apply(phi) as tf.SymbolicTensor;

    const phiOutput = createDenseLayer({
        name: Model.Policy + '_phi_output',
        units: CONFIG.gSDE.latentDim,
        useBias: true,
        activation: 'linear',
        kernelInitializer: 'orthogonal',
    }).apply(phiDetached) as tf.SymbolicTensor;

    const model = tf.model({
        name: Model.Policy,
        inputs: Object.values(inputs),
        outputs: [meanOutput, phiOutput], // [mean, phi]
    });
    model.optimizer = new PatchedAdamOptimizer(CONFIG.lrConfig.initial);
    model.loss = 'meanSquaredError'; // fake loss for save optimizer with model

    return model;
}

export function createValueNetwork(): tf.LayersModel {
    const { inputs, network } = createBaseNetwork(Model.Value, valueNetworkConfig);
    const valueOutput = createDenseLayer({
        name: Model.Value + '_output',
        units: 1,
        useBias: true,
        activation: 'linear',
    }).apply(network) as tf.SymbolicTensor;
    const model = tf.model({
        name: Model.Value,
        inputs: Object.values(inputs),
        outputs: valueOutput,
    });
    model.optimizer = new PatchedAdamOptimizer(CONFIG.lrConfig.initial);
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
            token: envToken,
            depth: 2,
            numHeads: config.heads,
        },
    );

    const flattenFinalToken = tf.layers.flatten({ name: modelName + '_flattenFinalToken' }).apply(transformedEnvToken) as tf.SymbolicTensor;
    const normFinalToken = tf.layers.layerNormalization({ name: modelName + '_normFinalToken' }).apply(flattenFinalToken) as tf.SymbolicTensor;

    const finalMLP = applyMLP(
        modelName + '_finalMLP',
        normFinalToken,
        config.finalMLP,
    );

    return { inputs, network: finalMLP, phi: normFinalToken };
}
