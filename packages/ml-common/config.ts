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

    policyEntropy: (iteration: number) => {
        return lerp(0.001, 0.01, clamp(1 - ((iteration - LEARNING_STEPS) / LEARNING_STEPS * 0.5), 0, 1));
    },

    minLogStd: (iteration: number) => {
        return -5; // lerp(-4, -2, clamp(iteration / (LEARNING_STEPS * 0.5), 0, 1))
        // Math.exp(-5) = 0.006738
    },
    maxLogStd: (iteration: number) => {
        return 0; // lerp(0, 2, clamp(iteration / (LEARNING_STEPS * 0.5), 0, 1))
        // Math.exp(0) = 1
    },

    policyEpochs: (iteration: number) => 3 - floor(clamp(iteration / (LEARNING_STEPS * 0.5), 0, 1) * 2),
    policyClipRatio: 0.2, // https://arxiv.org/pdf/2202.00079 - interesting idea don't clip at all

    valueEpochs: (iteration: number) => 3 - floor(clamp(iteration / (LEARNING_STEPS * 0.5), 0, 1) * 2),
    valueClipRatio: 0.2,
    valueLossCoeff: 0.5,

    // Dynamic learning rate adjustment based on KL
    lrConfig: {
        kl: {
            high: ACTION_DIM * 0.02,
            target: ACTION_DIM * 0.015,
            low: ACTION_DIM * 0.01,
        },
        initial: 1e-5,
        multHigh: 0.9,
        multLow: 1.05,
        min: 5e-6,
        max: 1e-3,
    },

    // Dynamic perturbation scale adjustment based on KL_noise
    perturbWeightsConfig: {
        kl: {
            high: 2 * ACTION_DIM * 0.02,
            target: 2 * ACTION_DIM * 0.015,
            low: 2 * ACTION_DIM * 0.01,
        },
        initial: 0.01,
        multHigh: 0.9,
        multLow: 1.05,
        min: 0.003,
        max: 0.02,
    },
    perturbChance: (iteration: number) => lerp(0.01, 0.7, clamp(iteration / (LEARNING_STEPS * 0.5), 0, 1)),

    batchSize: (iteration: number) => {
        return 1024 * clamp(ceil(6 * (iteration + 1) / (LEARNING_STEPS * 0.20)), 4, 16);
    },
    miniBatchSize: (iteration: number) => {
        return 64 * clamp(ceil((iteration + 1) / (LEARNING_STEPS * 0.25)), 1, 4);
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

// SAC configuration
export const SAC_CONFIG = {
    // Temperature (entropy coefficient)
    alpha: 0.2,                    // initial entropy coefficient (or use auto-tune)
    autoTuneAlpha: true,           // enable automatic alpha tuning
    targetEntropy: -ACTION_DIM,    // target entropy for auto-tuning
    alphaLR: 3e-4,                 // learning rate for alpha

    // Soft target update
    tau: 0.005,                    // Polyak averaging coefficient

    // Discount factor
    gamma: 0.99,                   // discount factor

    // Replay buffer
    replayBufferSize: 1_000_000,   // replay buffer size
    prioritizedReplay: false,      // use Prioritized Experience Replay (TODO: enable later)
    priorityAlpha: 0.6,            // prioritization exponent
    priorityBeta: 0.4,             // importance sampling exponent

    // Training
    batchSize: 256,                // batch size for training
    miniBatchSize: 256,            // mini-batch size (same as batch for now)
    actorUpdateFreq: 1,            // actor update frequency
    criticUpdateFreq: 1,           // critic update frequency
    targetUpdateFreq: 1,           // target network update frequency

    // Learning rates
    actorLR: 3e-4,                 // actor learning rate
    criticLR: 3e-4,                // critic learning rate

    // Gradient clipping
    clipNorm: 1.0,                 // gradient clipping norm

    // Exploration
    initialRandomSteps: 10000,     // initial random exploration steps

    // Log std bounds
    minLogStd: -20,                // minimum log std
    maxLogStd: 2,                  // maximum log std

    // Training parameters - FRAMES = Nsec / TICK_TIME_SIMULATION
    episodeFrames: Math.round(2 * 60 * 1000 / TICK_TIME_SIMULATION),

    // Workers
    workerCount: 8,
    backpressureQueueSize: 2,

    // Training control
    savePath: 'SAC_MHA',
};

// Current active experiment
export let CONFIG = DEFAULT_EXPERIMENT;
