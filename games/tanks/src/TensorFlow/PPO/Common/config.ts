// Experiment configuration for tank reinforcement learning with PPO
// This allows fine-tuning the RL model and experimenting with different hyperparameters

// Define experiment configurations that can be easily switched

export type Config = {
    name: string;
    // Learning parameters
    gamma: number;                  // Discount factor
    lam: number;                    // GAE lambda
    // PPO-specific parameters
    epochs: number;                // Number of epochs to train on the same data
    entropyCoeff: number;           // Entropy coefficient for encouraging exploration

    klConfig: {
        target: number,
        highCoef: number,
        lowCoef: number,
        max: number,
    },
    lrConfig: {
        initial: number,
        multHigh: number,
        multLow: number,
        min: number,
        max: number,
    },
    clipRatioConfig: {
        initial: number,
        deltaHigh: number,
        deltaLow: number,
        min: number,
        max: number,
    }

    trustCoeff: number;

    batchSize: number;              // Batch size for worker
    miniBatchSize: number,
    warmupFrames: number;              // Maximum number of frames to train on
    episodeFrames: number;              // Maximum number of frames to train on
    // Workers
    workerCount: number;                // Number of parallel workers
    reuseLimit: number;                 // Number of times a worker models can be reused without updating
    // Training control
    savePath: string;
};

// Default experiment configuration for PPO
export const DEFAULT_EXPERIMENT: Config = {
    name: 'ppo-default',
    // Learning parameters
    gamma: 0.99,
    lam: 0.95,
    // PPO-specific parameters
    epochs: 6,
    entropyCoeff: 0.01,

    klConfig: {
        target: 0.02,
        highCoef: 2.0,       // Если KL > 2 * 0.01 => 0.02
        lowCoef: 1 / 2.0,    // Если KL < 0.5 * 0.01 => 0.005
        max: 0.5,
    },
    lrConfig: {
        initial: 1e-4,
        multHigh: 0.9,
        multLow: 1.1,
        min: 1e-6,
        max: 5e-3,
    },
    clipRatioConfig: {
        initial: 0.2,
        deltaHigh: 0.005,
        deltaLow: 0.005,
        min: 0.05,
        max: 0.2,
    },

    trustCoeff: 0.1,

    batchSize: 256, // useless for appo
    miniBatchSize: 128,
    // Training parameters
    warmupFrames: 100,
    episodeFrames: 900, // usually produce 250 samples
    // Workers
    workerCount: 8,
    reuseLimit: 1,
    // Training control
    savePath: 'APPO_v1',
};

// Map of available experiments
export const EXPERIMENTS: { [key: string]: Config } = {
    default: DEFAULT_EXPERIMENT,
};

// Function to get experiment config by name
export function getExperimentConfig(name: string): Config {
    return EXPERIMENTS[name] || DEFAULT_EXPERIMENT;
}

// Current active experiment
export let CONFIG: Config = DEFAULT_EXPERIMENT;

// Set current experiment
export function setExperiment(nameOrConfig: string | Config): Config {
    if (typeof nameOrConfig === 'string') {
        CONFIG = getExperimentConfig(nameOrConfig);
    } else {
        CONFIG = nameOrConfig;
    }
    console.log(`Activated experiment: ${ CONFIG.name }`);
    return CONFIG;
}

// Get current experiment
export function getCurrentConfig(): Config {
    return CONFIG;
}
