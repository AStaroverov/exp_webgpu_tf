import { isMac } from '../../../../../lib/detect.ts';

export type Config = {
    // Learning parameters
    clipNorm: number;
    gamma: number;                  // Discount factor
    // PPO-specific parameters
    policyEpochs: number;                // Number of epochs to train on policy network
    valueEpochs: number;                  // Number of epochs to train the value network
    clipRatio: number;             // Clipping ratio for PPO
    entropyCoeff: number;           // Entropy coefficient for encouraging exploration

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
    warmupFrames: number;              // Maximum number of frames to train on
    episodeFrames: number;              // Maximum number of frames to train on
    // Workers
    workerCount: number;                // Number of parallel workers
    // Training control
    savePath: string;
    fsModelPath?: string;
};

// Default experiment configuration for PPO
export const DEFAULT_EXPERIMENT: Config = {
    // Learning parameters
    clipNorm: 5,
    gamma: 0.97,
    // PPO-specific parameters
    policyEpochs: 6,
    valueEpochs: 2,
    clipRatio: 0.2,
    entropyCoeff: isMac ? 0.01 : 0.001,

    klConfig: {
        target: 0.03,
        high: 0.04,
        low: 0.02,
        max: 0.5,
    },
    lrConfig: {
        initial: 1e-4,
        multHigh: 0.95,
        multLow: 1.05,
        min: 1e-6,
        max: 5e-3,
    },

    batchSize: isMac ? 500 : 2000,
    miniBatchSize: isMac ? 128 : 512,

    // Training parameters
    warmupFrames: 100,
    episodeFrames: 1200, // usually produce 250 samples
    // Workers
    workerCount: isMac ? 6 : 10,
    // Training control
    savePath: isMac ? 'APPO_VTRACE' : 'APPO_v1',
    // fsModelPath: 'v11-wo-vtrace',
};

// Current active experiment
export let CONFIG: Config = DEFAULT_EXPERIMENT;
