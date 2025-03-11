// Experiment configuration for tank reinforcement learning with PPO
// This allows fine-tuning the RL model and experimenting with different hyperparameters

// Define experiment configurations that can be easily switched
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';

export type RLExperimentConfig = {
    name: string;
    description: string;
    // Network architecture
    hiddenLayers: [ActivationIdentifier, number][];
    // Learning parameters
    learningRatePolicy: number;     // Learning rate for policy network
    learningRateValue: number;      // Learning rate for value network
    gamma: number;                  // Discount factor
    lam: number;                    // GAE lambda
    // PPO-specific parameters
    clipRatio: number;             // PPO clipping parameter
    epochs: number;                // Number of epochs to train on the same data
    entropyCoeff: number;           // Entropy coefficient for encouraging exploration
    // Training control
    saveModelEvery: number;         // Save model every N episodes
};

// Default experiment configuration for PPO
export const DEFAULT_EXPERIMENT: RLExperimentConfig = {
    name: 'ppo-default',
    description: 'Balanced PPO configuration for tank RL training',
    // Network architecture
    hiddenLayers: [['tanh', 256], ['tanh', 256], ['tanh', 128]],
    // Learning parameters
    learningRatePolicy: 0.0001,           // Generally smaller for PPO
    learningRateValue: 0.001,           // Generally smaller for PPO
    gamma: 0.99,
    lam: 0.97,                      // GAE lambda for advantage estimation
    // PPO-specific parameters
    clipRatio: 0.2,                // Standard PPO clipping parameter
    epochs: 5,                   // Train 4 epochs over the same data
    entropyCoeff: 0.01,             // Entropy regularization coefficient
    // Memory parameters
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
    console.log(`Description: ${ currentExperiment.description }`);
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