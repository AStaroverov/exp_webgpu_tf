import * as tf from '@tensorflow/tfjs';
import { CONFIG } from '../config.ts';
import {
    ACTION_HEAD_DIMS,
} from './dims.ts';

import { createDenseLayer } from "../../../ppo/src/models/ApplyLayers.ts";
import { Model } from '../../../ppo/src/models/def.ts';
import { createNetwork } from './Networks/v13.ts';
import { AdamW } from '../../../ppo/src/models/Optimizer/AdamW.ts';

export {
    RAY_HIT_TYPE_COUNT,
    TANK_FEATURES_DIM,
    TANK_HISTORY_STEPS,
    TANK_HISTORY_FEATURE_DIM,
    TURRET_SLOTS,
    TURRET_FEATURES_DIM,
    ENEMY_SLOTS,
    ENEMY_FEATURES_DIM,
    ALLY_SLOTS,
    ALLY_FEATURES_DIM,
    BULLET_SLOTS,
    BULLET_FEATURES_DIM,
    RAY_SLOTS,
    RAY_FEATURES_DIM,
    GRID_SIZE,
    GRID_CELL_FEATURES,
    GRID_CELLS,
    ACTION_HEAD_DIMS,
} from './dims.ts';

export { shouldNoiseLayer } from '../../../ppo/src/models/noiseGate.ts';

export function createPolicyNetwork(): tf.LayersModel {
    const {inputs, heads} = createNetwork(Model.Policy);

    // Create logits output for each head
    const logitsOutputs = heads.map((head, i) => {
        const units = ACTION_HEAD_DIMS[i];
        return createDenseLayer({
            name: Model.Policy + '_head_logits_' + i,
            units: units,
            useBias: true,
            activation: 'linear',
            biasInitializer: 'zeros',
            // kernelInitializer: tf.initializers.truncatedNormal({ mean: 0, stddev: 1 }),
            // kernelInitializer: tf.initializers.randomUniform({minval: -0.03, maxval: 0.03}),
            // noisy: true,
            // sigma: 0.03,
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
