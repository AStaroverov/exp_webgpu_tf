import * as tf from '@tensorflow/tfjs';
import { CONFIG } from '../../../ml-common/config.ts';
import {
    MAX_ALLIES,
    MAX_BULLETS,
    MAX_ENEMIES,
} from '../../../tanks/src/Pilots/Components/TankState.ts';

import { createDenseLayer } from "./ApplyLayers.ts";
import { Model } from './def.ts';
import { createNetwork } from './Networks/v9.ts';
import { AdamW } from './Optimizer/AdamW.ts';
import { NoisyDenseLayer } from './Layers/NoisyDenseLayer.ts';

export const CONTROLLER_FEATURES_DIM = 4;
export const BATTLE_FEATURES_DIM = 6;
export const TANK_FEATURES_DIM = 10;
export const ENEMY_SLOTS = MAX_ENEMIES;
export const ENEMY_FEATURES_DIM = 10;
export const ALLY_SLOTS = MAX_ALLIES;
export const ALLY_FEATURES_DIM = 10;
export const BULLET_SLOTS = MAX_BULLETS;
export const BULLET_FEATURES_DIM = 4;

export const ACTION_HEAD_DIMS = [15, 15, 2, 31];

export const shouldNoiseLayer = (name: string, _iteration?: number) => {
    name = name.toLowerCase();
    if (name.includes('head')) {
        return true;
    }

    if (name.includes('noisytransformer')) {
        return false;
    }

    return false;
}

export function createPolicyNetwork(): tf.LayersModel {
    const {inputs, heads} = createNetwork(Model.Policy);

    // Create logits output for each head
    const logitsOutputs = heads.map((head, i) => {
        const units = ACTION_HEAD_DIMS[i];
        return new NoisyDenseLayer({
            name: Model.Policy + '_head_logits_' + i,
            units: units,
            useBias: true,
            activation: 'linear',
            biasInitializer: 'zeros',
            sigma: 0.25,
        }).apply(head) as tf.SymbolicTensor;
    });

    const model = tf.model({
        name: Model.Policy,
        inputs: Object.values(inputs),
        outputs: logitsOutputs,
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
