import { clamp } from 'lodash';
import { ceil, lerp } from '../../../../lib/math.ts';
import { LEARNING_STEPS, TICK_TIME_SIMULATION } from '../Common/consts.ts';

export type Config = {
    // Learning parameters
    clipNorm: number;
    gamma: (iteration: number) => number;                  // Discount factor
    // PPO-specific parameters
    policyEpochs: number;                // Number of epochs to train on policy network
    policyClipRatio: number;             // Clipping ratio for PPO
    policyEntropy: (iteration: number) => number

    valueEpochs: number;                  // Number of epochs to train the value network
    valueClipRatio: number;             // Clipping ratio for PPO
    valueLossCoeff: number;                  // Number of epochs to train the value network

    klConfig: {
        target: number,
        high: number,
        low: number,
        max: number,
    },
    lrConfig: {
        initial: number,
        multHigh: number,
        multLow: number,
        min: number,
        max: number,
    },

    batchSize: (iteration: number) => number;              // Batch size for worker
    miniBatchSize: (iteration: number) => number,
    episodeFrames: number;              // Maximum number of frames to train on
    // Workers
    workerCount: number;
    backpressureQueueSize: number;          // Number of batches in the queue before applying backpressure
    // Perturbation of weights
    perturbChance: (iteration: number) => number;
    perturbWeightsScale: (iteration: number) => number;
    // Training control
    savePath: string;
    fsModelPath?: string;
};

// Default experiment configuration for PPO
export const DEFAULT_EXPERIMENT: Config = {
    // Learning parameters
    clipNorm: 20,
    // PPO-specific parameters
    gamma: (iteration) => {
        return lerp(0.92, 0.997, clamp(iteration / (LEARNING_STEPS * 2.5), 0, 1))
    },
    policyEpochs: 1,
    policyClipRatio: 0.2,
    policyEntropy: (iteration, min = 0.01, max = 0.05, totalIterations = LEARNING_STEPS * 5) => {
        const k = Math.log((max - min) / 1e-3) / totalIterations;
        return min + (max - min) * Math.exp(-k * iteration);
    },

    valueEpochs: 1,
    valueClipRatio: 0.4,
    valueLossCoeff: 0.5,

    klConfig: {
        target: 0.01,
        high: 0.02,
        low: 0.005,
        max: 0.5,
    },
    lrConfig: {
        initial: 1e-5,
        multHigh: 0.95,
        multLow: 1.05,
        min: 1e-6,
        max: 1e-3,
    },

    batchSize: (iteration) => {
        return (4 * 1024) * clamp(ceil(iteration / (LEARNING_STEPS / 2)), 1, 4);
    },
    miniBatchSize: (iteration) => {
        return 64 * clamp(ceil(iteration / (LEARNING_STEPS / 2)), 1, 4);
    },

    // Training parameters - FRAMES = Nsec / TICK_TIME_SIMULATION
    episodeFrames: Math.round(2 * 60 * 1000 / TICK_TIME_SIMULATION),
    // Workers
    workerCount: 8,
    backpressureQueueSize: 2,
    // Perturbation of weights
    perturbChance: (iteration) => lerp(0.01, 0.10, clamp(1 - iteration / LEARNING_STEPS, 0, 1)),
    perturbWeightsScale: (iteration) => lerp(0.005, 0.02, clamp(1 - iteration / LEARNING_STEPS, 0, 1)),
    // Training control
    savePath: 'PPO_MHA',
    fsModelPath: '/assets/models/v1',
};

// Current active experiment
export let CONFIG: Config = DEFAULT_EXPERIMENT;
