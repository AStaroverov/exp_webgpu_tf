import { clamp } from 'lodash';
import { lerp } from '../../lib/math.ts';
import { LEARNING_STEPS, TICK_TIME_SIMULATION } from './consts.ts';

export const CONFIG = {
    clipNorm: 5,

    gamma: (iteration: number) => {
        return 0.97;//lerp(0.97, 0.997, clamp(iteration / LEARNING_STEPS, 0, 1))
    },

    policyEntropy: (iteration: number) => {
        return lerp(0.01, 0.1, 1 - clamp(iteration / LEARNING_STEPS, 0, 1));
    },

    policyEpochs: (_iter: number) => 4,
    policyClipRatio: 0.6,

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
        initial: 1e-4,
        multHigh: 0.95,
        multLow: 1.01,
        min: 1e-5,
        max: 1e-3,
    },

    batchSize: (iteration: number) => {
        return 256 * 16;
    },
    miniBatchSize: (iteration: number) => {
        return 256;
    },

    // Training parameters - FRAMES = Nsec / TICK_TIME_SIMULATION
    episodeFrames: Math.round(60 * 1000 / TICK_TIME_SIMULATION),
    // Workers
    workerCount: 4,
    backpressureQueueSize: 2,
    // Training control
    savePath: 'PPO_MHA',
    // fsModelPath: '/assets/models/v1',
};
