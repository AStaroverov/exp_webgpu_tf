import { isMac } from '../../../../../lib/detect.ts';
import { TICK_TIME_SIMULATION } from '../Common/consts.ts';

export type Config = {
    // Learning parameters
    clipNorm: number;
    gamma: number;                  // Discount factor
    // PPO-specific parameters
    policyEpochs: number;                // Number of epochs to train on policy network
    policyClipRatio: number;             // Clipping ratio for PPO
    policyEntropyCoeff: number;           // Entropy coefficient for encouraging exploration

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
    gamma: 0.92,
    policyEpochs: 2,
    policyClipRatio: 0.2,
    policyEntropyCoeff: 0.01,

    valueEpochs: 2,
    valueClipRatio: 0.4,
    valueLossCoeff: 0.5,

    klConfig: {
        target: 0.01,
        high: 0.02,
        low: 0.0075,
        max: 0.5,
    },
    lrConfig: {
        initial: 1e-5,
        multHigh: 0.95,
        multLow: 1.05,
        min: 1e-5,
        max: 1e-3,
    },

    batchSize: isMac ? 200 : 3_000,
    miniBatchSize: isMac ? 128 : 128,

    // Training parameters - FRAMES = Nsec / TICK_TIME_SIMULATION
    episodeFrames: Math.round(40 * 1000 / TICK_TIME_SIMULATION),
    // Workers
    workerCount: isMac ? 3 : 9,
    // perturbWeights
    perturbWeightsScale: 0.02,
    // Training control
    savePath: isMac ? 'PPO_MHA' : 'PPO_MHA_V1',
    // fsModelPath: 'v20',
};

// Current active experiment
export let CONFIG: Config = DEFAULT_EXPERIMENT;
