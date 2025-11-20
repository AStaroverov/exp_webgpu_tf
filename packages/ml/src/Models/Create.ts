import * as tf from '@tensorflow/tfjs';
import {CONFIG} from '../../../ml-common/config.ts';
import {
    ALLY_BUFFER,
    BULLET_BUFFER,
    ENEMY_BUFFER,
    MAX_ALLIES,
    MAX_BULLETS,
    MAX_ENEMIES,
} from '../../../tanks/src/Pilots/Components/TankState.ts';

import {ACTION_DIM} from '../../../ml-common/consts.ts';
import {Model} from './def.ts';
import {LogStdLayer} from './Layers/LogStdLayer.ts';
import {createNetwork} from './Networks/v3.ts';
import {AdamW} from './Optimizer/AdamW.ts';
import {createDenseLayer} from "./ApplyLayers.ts";

export const CONTROLLER_FEATURES_DIM = 4;
export const BATTLE_FEATURES_DIM = 6;
export const TANK_FEATURES_DIM = 7;
export const ENEMY_SLOTS = MAX_ENEMIES;
export const ENEMY_FEATURES_DIM = ENEMY_BUFFER - 1; // -1 потому что id не считаем
export const ALLY_SLOTS = MAX_ALLIES;
export const ALLY_FEATURES_DIM = ALLY_BUFFER - 1; // -1 потому что id не считаем
export const BULLET_SLOTS = MAX_BULLETS;
export const BULLET_FEATURES_DIM = BULLET_BUFFER - 1; // -1 потому что id не считаем

export function createPolicyNetwork(): tf.LayersModel {
    const {inputs, heads, latent} = createNetwork(Model.Policy);

    const means = heads.map((head, i) => {
        const item = createDenseLayer({
            name: Model.Policy + '_head_output' + '_' + i,
            units: 1,
            useBias: false,
            activation: 'tanh',
            biasInitializer: 'zeros',
            kernelInitializer: tf.initializers.randomNormal({mean: 0, stddev: 0.02}),
        }).apply(head) as tf.SymbolicTensor;

        return item;
    });

    // const meanOutput = createDenseLayer({
    //     name: Model.Policy + '_mean_output',
    //     units: ACTION_DIM,
    //     useBias: true,
    //     activation: 'linear',
    //     biasInitializer: 'zeros',
    //     kernelInitializer: tf.initializers.randomNormal({mean: 0, stddev: 0.02}),
    // }).apply(lan) as tf.SymbolicTensor;

    const logStdOutput = new LogStdLayer({
        name: Model.Policy + '_log_std_output',
        units: ACTION_DIM
    }).apply(latent) as tf.SymbolicTensor;

    const model = tf.model({
        name: Model.Policy,
        inputs: Object.values(inputs),
        outputs: [logStdOutput, ...means], // [mean, phi]
    });
    model.optimizer = new AdamW(CONFIG.lrConfig.initial);
    model.loss = 'meanSquaredError'; // fake loss for save optimizer with model

    return model;
}

export function createValueNetwork(): tf.LayersModel {
    const {inputs, heads} = createNetwork(Model.Value);
    const valueOutput = createDenseLayer({
        name: Model.Value + '_output',
        units: 1,
        useBias: true,
        activation: 'linear',
        biasInitializer: 'zeros',
        kernelInitializer: 'glorotUniform',
    }).apply(heads[0]) as tf.SymbolicTensor;
    const model = tf.model({
        name: Model.Value,
        inputs: Object.values(inputs),
        outputs: valueOutput,
    });
    model.optimizer = new AdamW(CONFIG.lrConfig.initial);
    model.loss = 'meanSquaredError';

    return model;
}
