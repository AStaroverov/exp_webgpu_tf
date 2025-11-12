import { clamp } from 'lodash';
import { ceil, lerp } from '../../lib/math.ts';
import { ACTION_DIM, LEARNING_STEPS, TICK_TIME_SIMULATION } from './consts.ts';

export const CONFIG = {
    clipNorm: 1,

    gamma: (iteration: number) => {
        return lerp(0.97, 0.997, clamp(iteration / LEARNING_STEPS, 0, 1))
    },

    policyEntropy: (iteration: number) => {
        return lerp(0.0001, 0.01, clamp(1 - ((iteration - LEARNING_STEPS) / LEARNING_STEPS * 0.5), 0, 1)) / ACTION_DIM;
    },

    minLogStd: (iteration: number) => {
        return [
            -(5), // + clamp(iteration / (LEARNING_STEPS * 0.3), 0, 1) * 2
            -(5), // + clamp(iteration / (LEARNING_STEPS * 0.3), 0, 1) * 2
            -(5), // + clamp(iteration / (LEARNING_STEPS * 0.3), 0, 1) * 2
            -(5), // + clamp(iteration / (LEARNING_STEPS * 0.3), 0, 1) * 2
        ]
    },
    maxLogStd: (iteration: number) => {
        return [
            -(0.8),// + clamp(iteration / (LEARNING_STEPS * 0.3), 0, 1) * 1.5),
            -(0.8),// + clamp(iteration / (LEARNING_STEPS * 0.3), 0, 1) * 1.5),
            -(0.8),// + clamp(iteration / (LEARNING_STEPS * 0.3), 0, 1) * 1.5),
            -(0.8 + clamp(iteration / (LEARNING_STEPS * 0.3), 0, 1) * 1),
            // -(1 + clamp(iteration / (LEARNING_STEPS * 0.2), 0, 1) * 1.5)
        ]
    },

    policyEpochs: (iteration: number) => 2,
    policyClipRatio: 0.2,

    valueEpochs: (iteration: number) => 2,
    valueClipRatio: 0.2,
    valueLossCoeff: 0.5,
    valueLRCoeff: 1,

    // Dynamic learning rate adjustment based on KL
    lrConfig: {
        kl: {
            high: ACTION_DIM * 0.023,
            target: ACTION_DIM * 0.02,
            low: ACTION_DIM * 0.017,
        },
        initial: 1e-5,
        multHigh: 0.95,
        multLow: 1.01,
        min: 1e-6,
        max: 1e-3,
    },

    batchSize: (iteration: number) => {
        return 1024 * clamp(ceil(6 * (iteration + 1) / (LEARNING_STEPS * 0.20)), 4, 16);
    },
    miniBatchSize: (iteration: number) => {
        return 64 * clamp(ceil((iteration + 1) / (LEARNING_STEPS * 0.25)), 2, 4);
    },

    // Training parameters - FRAMES = Nsec / TICK_TIME_SIMULATION
    episodeFrames: Math.round(2 * 60 * 1000 / TICK_TIME_SIMULATION),
    // Workers
    workerCount: 6,
    backpressureQueueSize: 2,
    // Training control
    savePath: 'PPO_MHA',
    // fsModelPath: '/assets/models/v1',
};
