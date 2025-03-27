// Experiment configuration for tank reinforcement learning with PPO
// This allows fine-tuning the RL model and experimenting with different hyperparameters

// Define experiment configurations that can be easily switched
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';

export type Config = {
    name: string;
    // Network architecture
    hiddenLayers: [ActivationIdentifier, number][];
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

    batchSize: number;              // Batch size for worker
    miniBatchSize: number,
    warmupFrames: number;              // Maximum number of frames to train on
    episodeFrames: number;              // Maximum number of frames to train on
    // Workers
    workerCount: number;                // Number of parallel workers
    // Training control
    saveModelEvery: number;         // Save model every N episodes
    savePath: string;
};

// Default experiment configuration for PPO
export const DEFAULT_EXPERIMENT: Config = {
    name: 'ppo-default',
    // Network architecture
    hiddenLayers: [['relu', 256], ['relu', 256], ['relu', 128], ['relu', 64]],
    // Learning parameters
    learningRatePolicy: 1e-4,
    learningRateValue: 1e-4,
    gamma: 0.99,
    lam: 0.95,
    // PPO-specific parameters
    epochs: 6,
    clipRatioPolicy: 0.2,
    clipRatioValue: 0.25,
    entropyCoeff: 0.01,
    maxKL: 0.05,

    batchSize: 256, // useless for appo
    miniBatchSize: 256,
    warmupFrames: 100,
    episodeFrames: 900, // usually produce 250 samples
    // Workers
    workerCount: 12,
    // Training control
    saveModelEvery: 1,
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
