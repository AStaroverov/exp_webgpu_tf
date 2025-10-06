import { clamp, floor } from 'lodash';
import { ceil, lerp } from '../../lib/math.ts';
import { random } from '../../lib/random.ts';
import { LEARNING_STEPS, TICK_TIME_SIMULATION } from './consts.ts';

// Default experiment configuration for PPO
export const DEFAULT_EXPERIMENT = {
    // Learning parameters
    clipNorm: 20,
    // PPO-specific parameters
    gamma: (iteration: number) => {
        return lerp(0.95, 0.997, clamp(iteration / LEARNING_STEPS, 0, 1))
    },
    policyEpochs: (iteration: number) => 3 - floor(clamp(iteration / (LEARNING_STEPS * 0.2), 0, 1) * 2),
    policyClipRatio: 0.2,
    policyEntropy: (iteration: number) => {
        return lerp(0.005, 0.05, clamp(1 - iteration / (LEARNING_STEPS * 0.2), 0, 1))
    },

    valueEpochs: (iteration: number) => 3 - floor(clamp(iteration / (LEARNING_STEPS * 0.5), 0, 1) * 2),
    valueClipRatio: 0.2,
    valueLossCoeff: 0.5,

    klConfig: {
        target: 0.01,
        high: 0.02,
        low: 0.005,
        max: 5,
    },
    lrConfig: {
        initial: 1e-5,
        multHigh: 0.95,
        multLow: 1.05,
        min: 1e-6,
        max: 1e-3,
    },

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
    // Perturbation of weights
    perturbChance: (iteration: number) => lerp(0.01, 0.1, clamp(iteration / (LEARNING_STEPS * 0.2), 0, 1)),
    perturbWeightsScale: (iteration: number) => 0.001 + random() * lerp(0, 0.01, clamp(iteration / (LEARNING_STEPS * 0.2), 0, 1)),
    // Training control
    savePath: 'PPO_MHA',
    // fsModelPath: '/assets/models/v1',
};

// Current active experiment
export let CONFIG = DEFAULT_EXPERIMENT;
