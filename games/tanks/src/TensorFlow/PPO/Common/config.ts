// Experiment configuration for tank reinforcement learning with PPO
// This allows fine-tuning the RL model and experimenting with different hyperparameters

// Define experiment configurations that can be easily switched

export type Config = {
    name: string;
    // Learning parameters
    learningRatePolicy: number;     // Learning rate for policy network
    learningRateValue: number;      // Learning rate for value network
    gamma: number;                  // Discount factor
    lam: number;                    // GAE lambda
    // PPO-specific parameters
    clipRatioPolicy: number;             // PPO clipping parameter
    clipRatioValue: number;             // PPO clipping parameter
    epochs: number;                // Number of epochs to train on the same data
    entropyCoeff: number;           // Entropy coefficient for encouraging exploration
    maxKL: number;                  // Maximum KL divergence between old and new policy

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
    learningRatePolicy: 5e-4,
    learningRateValue: 5e-4,
    gamma: 0.99,
    lam: 0.95,
    // PPO-specific parameters
    epochs: 8,
    clipRatioPolicy: 0.2,
    clipRatioValue: 0.2,
    entropyCoeff: 0.005,
    maxKL: 0.05,

    trustCoeff: 0.2,

    batchSize: 256, // useless for appo
    miniBatchSize: 256,
    warmupFrames: 100,
    episodeFrames: 900, // usually produce 250 samples
    // Workers
    workerCount: 8,
    reuseLimit: 0,
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
