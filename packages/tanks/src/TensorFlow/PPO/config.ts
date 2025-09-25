import { LEARNING_STEPS, TICK_TIME_SIMULATION } from '../Common/consts.ts';

export type Config = {
    // Learning parameters
    clipNorm: number;
    gamma: number;                  // Discount factor
    // PPO-specific parameters
    policyEpochs: number;                // Number of epochs to train on policy network
    policyClipRatio: number;             // Clipping ratio for PPO
    policyEntropy: {
        coeff: number;
        limit: number;
        reset: number;
    }

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

    batchSize: number;              // Batch size for worker
    miniBatchSize: number,
    episodeFrames: number;              // Maximum number of frames to train on
    // Workers
    workerCount: number;                // Number of parallel workers
    backpressureQueueSize: number;          // Number of batches in the queue before applying backpressure
    // perturbWeights
    perturbWeightsScale: number;
    // Training control
    savePath: string;
    fsModelPath?: string;
};

// Default experiment configuration for PPO
export const DEFAULT_EXPERIMENT: Config = {
    // Learning parameters
    clipNorm: 20,
    // PPO-specific parameters
    gamma: 0.95,
    policyEpochs: 2,
    policyClipRatio: 0.2,
    policyEntropy: {
        coeff: 0.025,
        limit: (LEARNING_STEPS * 0.3) / 4,
        reset: LEARNING_STEPS / 4,
    },

    valueEpochs: 2,
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
        min: 1e-5,
        max: 1e-3,
    },

    batchSize: 128 * 28, // isMac ? 200 : 3_000,
    miniBatchSize: 128, // isMac ? 128 : 128,

    // Training parameters - FRAMES = Nsec / TICK_TIME_SIMULATION
    episodeFrames: Math.round(60 * 1000 / TICK_TIME_SIMULATION),
    // Workers
    workerCount: 6, //isMac ? 2 : 8,
    backpressureQueueSize: 2,
    // perturbWeights
    perturbWeightsScale: 0.02,
    // Training control
    savePath: 'PPO_MHA', // isMac ? 'PPO_MHA' : 'PPO_MHA_V1',
    // fsModelPath: '/assets/models/v32',
};

// Current active experiment
export let CONFIG: Config = DEFAULT_EXPERIMENT;
