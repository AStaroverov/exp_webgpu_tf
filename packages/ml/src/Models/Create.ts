import * as tf from '@tensorflow/tfjs';
import { CONFIG } from '../../../ml-common/config.ts';
import {
    MAX_ALLIES,
    MAX_BULLETS,
    MAX_ENEMIES,
    RAYS_COUNT,
    RAY_HIT_TYPE_COUNT,
    RAY_BUFFER,
    MAX_TURRETS,
} from '../../../tanks/src/Pilots/Components/TankState.ts';


import { createDenseLayer } from "./ApplyLayers.ts";
import { Model } from './def.ts';
import { createNetwork } from './Networks/v10.ts';
import { AdamW } from './Optimizer/AdamW.ts';

export { RAY_HIT_TYPE_COUNT }; // LightTank, MediumTank, HeavyTank, PlayerTank, Harvester, MeleeCar

export const BATTLE_FEATURES_DIM = 2;
export const TANK_FEATURES_DIM = 8;

export const TURRET_SLOTS = MAX_TURRETS;
export const TURRET_FEATURES_DIM = 4;

// Enemies: [hp, x, y, vx, vy, turretRotationCos, turretRotationSin, colliderRadius]
export const ENEMY_SLOTS = MAX_ENEMIES;
export const ENEMY_FEATURES_DIM = 8 + 1; // 1 type embedding

// Allies: [hp, x, y, vx, vy, turretRotationCos, turretRotationSin, colliderRadius]
export const ALLY_SLOTS = MAX_ALLIES;
export const ALLY_FEATURES_DIM = 8 + 1; // 1 type embedding

// Bullets: [x, y, vx, vy]
export const BULLET_SLOTS = MAX_BULLETS;
export const BULLET_FEATURES_DIM = 4;

// Unified rays (environment + turret rays combined)
export const RAY_SLOTS = RAYS_COUNT;
export const RAY_FEATURES_DIM = RAY_BUFFER - 1 + 2; // [locRootX, locRootY, locDirX, locDirY, distance, hitObstacle, hitVehicle]

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
