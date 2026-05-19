import type { PpoConfig } from '../../ppo/src/config.ts';
import { TICK_TIME_SIMULATION } from './consts.ts';

export const CONFIG: PpoConfig & {
    episodeFrames: number;
    workerCount: number;
} = {
    clipNorm: 1,

    gamma: (_iteration: number) => {
        return 0.99;
    },

    adaptiveEntropy: {
        targetRatio: 0.5,          // target entropy as fraction of max entropy
        alphaLR: 0.01,             // learning rate for log_alpha update (per training pass)
        initialLogAlpha: Math.log(0.1),   // initial α ≈ 0.1
        minLogAlpha: Math.log(0.005),     // min α ≈ 0.005
        maxLogAlpha: Math.log(0.5),       // max α ≈ 0.5
    },

    policyEpochs: (_iter: number) => 4,
    policyClipRatio: 0.2,

    valueEpochs: (_iter: number) => 4,
    valueClipRatio: 0.2,
    valueLossCoeff: 0.5,
    valueLRCoeff: 1,

    // Dynamic learning rate adjustment based on KL
    lrConfig: {
        kl: {
            high: 0.013,
            target: 0.01,
            low: 0.007,
        },
        initial: 1e-5,
        multHigh: 0.95,
        multLow: 1.01,
        min: 1e-5,
        max: 1e-3,
    },

    batchSize: (_iteration: number) => {
        return 256 * 16;
    },
    miniBatchSize: (_iteration: number) => {
        return 256;
    },

    // Training parameters - FRAMES = Nsec / TICK_TIME_SIMULATION
    episodeFrames: Math.round(3 * 60 * 1000 / TICK_TIME_SIMULATION),
    // Workers
    workerCount: 4,
    backpressureQueueSize: 2,
    // Training control
    savePath: 'PPO_MHA',
    // fsModelPath: '/assets/models/v1',
};
