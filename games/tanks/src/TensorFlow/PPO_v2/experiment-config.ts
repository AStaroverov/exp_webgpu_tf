// Experiment configuration for tank reinforcement learning with PPO
// This allows fine-tuning the RL model and experimenting with different hyperparameters

// Define experiment configurations that can be easily switched
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';

export type RLExperimentConfig = {
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
    batchSize: number;              // Batch size for training
    epochs: number;                // Number of epochs to train on the same data
    entropyCoeff: number;           // Entropy coefficient for encouraging exploration
    // Training control
    saveModelEvery: number;         // Save model every N episodes
};

// Default experiment configuration for PPO
export const DEFAULT_EXPERIMENT: RLExperimentConfig = {
    name: 'ppo-default',
    // Network architecture
    hiddenLayers: [['relu', 256], ['relu', 128], ['relu', 128], ['relu', 64]],
    // Learning parameters
    learningRatePolicy: 1e-4,
    learningRateValue: 1e-5,
    gamma: 0.95,
    lam: 0.95,
    // PPO-specific parameters
    epochs: 14,
    batchSize: 256,
    clipRatioPolicy: 0.15,
    clipRatioValue: 0.25,
    entropyCoeff: 0.05,
    // Training control
    saveModelEvery: 10,
};


// Map of available experiments
export const EXPERIMENTS: { [key: string]: RLExperimentConfig } = {
    default: DEFAULT_EXPERIMENT,
};

// Function to get experiment config by name
export function getExperimentConfig(name: string): RLExperimentConfig {
    return EXPERIMENTS[name] || DEFAULT_EXPERIMENT;
}

// Current active experiment
let currentExperiment: RLExperimentConfig = DEFAULT_EXPERIMENT;

// Set current experiment
export function setExperiment(nameOrConfig: string | RLExperimentConfig): RLExperimentConfig {
    if (typeof nameOrConfig === 'string') {
        currentExperiment = getExperimentConfig(nameOrConfig);
    } else {
        currentExperiment = nameOrConfig;
    }
    console.log(`Activated experiment: ${ currentExperiment.name }`);
    return currentExperiment;
}

// Get current experiment
export function getCurrentExperiment(): RLExperimentConfig {
    return currentExperiment;
}

// Export experiment settings for logging
export function getExperimentSettings(): string {
    return JSON.stringify(currentExperiment, null, 2);
}