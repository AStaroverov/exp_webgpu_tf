import * as tf from '@tensorflow/tfjs';
import { CONFIG } from '../../../ml-common/config.ts';
import {
    MAX_ALLIES,
    MAX_BULLETS,
    MAX_ENEMIES,
    ENV_RAYS_TOTAL,
    TURRET_RAYS_COUNT,
    RAY_HIT_TYPE_COUNT,
} from '../../../tanks/src/Pilots/Components/TankState.ts';


import { createDenseLayer } from "./ApplyLayers.ts";
import { Model } from './def.ts';
import { createNetwork } from './Networks/v9.ts';
import { AdamW } from './Optimizer/AdamW.ts';

export { RAY_HIT_TYPE_COUNT }; // LightTank, MediumTank, HeavyTank, PlayerTank, Harvester, MeleeCar

export const BATTLE_FEATURES_DIM = 2;
export const TANK_FEATURES_DIM = 10;

// Enemies: [hp, x, y]
export const ENEMY_SLOTS = MAX_ENEMIES;
export const ENEMY_FEATURES_DIM = 3;

// Allies: [hp, x, y]
export const ALLY_SLOTS = MAX_ALLIES;
export const ALLY_FEATURES_DIM = 3;

// Bullets: [x, y, vx, vy]
export const BULLET_SLOTS = MAX_BULLETS;
export const BULLET_FEATURES_DIM = 4;

// Environment rays: [locRayDirX, locRayDirY, locX, locY, radius, distance] = 6 features per ray + hitType for embedding
export const ENV_RAY_SLOTS = ENV_RAYS_TOTAL;
export const ENV_RAY_FEATURES_DIM = 6;

// Turret rays: [locRayDirX, locRayDirY, locX, locY, locVx, locVy, radius, distance, aimingError] = 9 features per ray + hitType for embedding
export const TURRET_RAY_SLOTS = TURRET_RAYS_COUNT;
export const TURRET_RAY_FEATURES_DIM = 9;

export const ACTION_HEAD_DIMS = [15, 15, 2, 31];

export const shouldNoiseLayer = (_name: string, _iteration?: number) => {
    return true;
}

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
            kernelInitializer: tf.initializers.randomUniform({minval: -0.03, maxval: 0.03}), 
            noisy: true,
            sigma: 0.03,
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
