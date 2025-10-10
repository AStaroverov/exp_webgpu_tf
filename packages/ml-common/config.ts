import { clamp, floor } from 'lodash';
import { ceil, lerp } from '../../lib/math.ts';
import { ACTION_DIM, LEARNING_STEPS, TICK_TIME_SIMULATION } from './consts.ts';

// Default experiment configuration for PPO
export const DEFAULT_EXPERIMENT = {
    // Learning parameters
    clipNorm: 1.0,
    // PPO-specific parameters
    gamma: (iteration: number) => {
        return lerp(0.95, 0.997, clamp(iteration / LEARNING_STEPS, 0, 1))
    },

    policyEpochs: (iteration: number) => 3 - floor(clamp(iteration / (LEARNING_STEPS * 0.5), 0, 1) * 2),
    policyClipRatio: 0.2, // https://arxiv.org/pdf/2202.00079 - interesting idea don't clip at all
    policyEntropy: (iteration: number) => {
        return lerp(0.01, 0.1, clamp(1 - iteration / (LEARNING_STEPS * 0.4), 0, 1))
    },

    valueEpochs: (iteration: number) => 3 - floor(clamp(iteration / (LEARNING_STEPS * 0.5), 0, 1) * 2),
    valueClipRatio: 0.2,
    valueLossCoeff: 0.5,

    // Dynamic learning rate adjustment based on KL
    lrConfig: {
        kl: {
            high: ACTION_DIM * 0.013,
            target: ACTION_DIM * 0.01,
            low: ACTION_DIM * 0.007,
        },
        initial: 1e-5,
        multHigh: 0.95,
        multLow: 1.05,
        min: 5e-6,
        max: 1e-3,
    },

    // Dynamic perturbation scale adjustment based on KL_noise
    perturbWeightsConfig: {
        kl: {
            target: 0.1,
            high: 0.2,
            low: 0.05,
        },
        initial: 0.01,
        multHigh: 0.95,
        multLow: 1.05,
        min: 0.003,
        max: 0.02,
    },
    perturbChance: (iteration: number) => lerp(0.01, 0.1, clamp(iteration / (LEARNING_STEPS * 0.2), 0, 1)),

    batchSize: (iteration: number) => {
        return (1024 * 4) * clamp(ceil(iteration / (LEARNING_STEPS * 0.5)), 1, 4);
    },
    miniBatchSize: (iteration: number) => {
        return 64 * clamp(ceil(iteration / (LEARNING_STEPS * 0.5)), 1, 4);
    },

    // Training parameters - FRAMES = Nsec / TICK_TIME_SIMULATION
    episodeFrames: Math.round(2 * 60 * 1000 / TICK_TIME_SIMULATION),
    // Workers
    workerCount: 8,
    backpressureQueueSize: 2,
    // Training control
    savePath: 'PPO_MHA',
    // fsModelPath: '/assets/models/v1',
};

// Current active experiment
export let CONFIG = DEFAULT_EXPERIMENT;
