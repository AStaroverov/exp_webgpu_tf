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
    createDenseLayer
} from './ApplyLayers.ts';

import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';
import { ACTION_DIM } from '../../../ml-common/consts.ts';
import { Model } from './def.ts';
import { LogStdLayer } from './Layers/LogStdLayer.ts';
import { StopGradientLayer } from './Layers/StopGradientLayer.ts';
import { createNetwork } from './Networks/v2.ts';
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

type policyNetworkConfig = NetworkConfig

const policyNetworkConfig: policyNetworkConfig = {
    dim: 64,
    heads: 2,
    finalMLP: [
        ['relu', 512],
        ['relu', 256],
        ['relu', 64],
    ],
};
const valueNetworkConfig: NetworkConfig = {
    dim: 16,
    heads: 1,
    finalMLP: [
        ['relu', 64],
        ['relu', 16],
    ] as [ActivationIdentifier, number][],
};

export function createPolicyNetwork(): tf.LayersModel {
    const { inputs, network, phi } = createNetwork(Model.Policy, policyNetworkConfig);

    const meanOutput = createDenseLayer({
        name: Model.Policy + '_mean_output',
        units: ACTION_DIM,
        useBias: true,
        activation: 'linear',
        biasInitializer: 'zeros',
        kernelInitializer: 'glorotUniform',
    }).apply(network) as tf.SymbolicTensor;

    const logStdOutput = new LogStdLayer({
        name: Model.Policy + '_log_std_output',
        units: ACTION_DIM
    }).apply(meanOutput) as tf.SymbolicTensor;

    const phiDetached = new StopGradientLayer({
        name: Model.Policy + '_phi_stop_gradient'
    }).apply(phi) as tf.SymbolicTensor;

    const phiOutput = createDenseLayer({
        name: Model.Policy + '_phi_dense',
        units: CONFIG.gSDE.latentDim,
        useBias: false,
        activation: 'tanh',
        kernelInitializer: 'orthogonal',
    }).apply(phiDetached) as tf.SymbolicTensor;

    const model = tf.model({
        name: Model.Policy,
        inputs: Object.values(inputs),
        outputs: [meanOutput, logStdOutput, phiOutput], // [mean, phi]
    });
    model.optimizer = new PatchedAdamOptimizer(CONFIG.lrConfig.initial);
    model.loss = 'meanSquaredError'; // fake loss for save optimizer with model

    return model;
}

export function createValueNetwork(): tf.LayersModel {
    const { inputs, network } = createNetwork(Model.Value, valueNetworkConfig);
    const valueOutput = createDenseLayer({
        name: Model.Value + '_output',
        units: 1,
        useBias: true,
        activation: 'linear',
        biasInitializer: 'zeros',
        kernelInitializer: 'zeros',
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
