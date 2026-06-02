/**
 * ppo_unknown CONFIG — a `PpoConfig` instance consumed unchanged by the generic
 * learner/actor in `packages/ppo`. Started from `ppo_tanks/src/config.ts`; only
 * `savePath` and episode sizing diverge for the hex game.
 */

import type { PpoConfig } from '../../ppo/src/config.ts';
import { TICK_TIME_SIMULATION } from './consts.ts';

export const CONFIG: PpoConfig & {
    episodeFrames: number;
    workerCount: number;
} = {
    clipNorm: 1,

    gamma: (_iteration: number) => 0.99,

    adaptiveEntropy: {
        targetRatio: 0.5,
        alphaLR: 0.01,
        initialLogAlpha: Math.log(0.1),
        minLogAlpha: Math.log(0.005),
        maxLogAlpha: Math.log(0.5),
    },

    policyEpochs: (_iter: number) => 4,
    policyClipRatio: 0.2,

    valueEpochs: (_iter: number) => 4,
    valueClipRatio: 0.2,
    valueLossCoeff: 0.5,
    valueLRCoeff: 1,

    lrConfig: {
        kl: { high: 0.013, target: 0.01, low: 0.007 },
        initial: 1e-3,
        multHigh: 0.95,
        multLow: 1.01,
        min: 1e-5,
        max: 1e-3,
    },

    batchSize: (_iteration: number) => 256 * 16,
    miniBatchSize: (_iteration: number) => 256,

    // Decision-based episodes are short in STEPS but long in TICKS. Cap ticks at
    // a few simulated minutes; termination is owned by ppo_unknown (no win cond
    // baked into the game).
    episodeFrames: Math.round(2 * 60 * 1000 / TICK_TIME_SIMULATION),
    workerCount: 4,
    backpressureQueueSize: 2,
    savePath: 'PPO_UNKNOWN',
};
