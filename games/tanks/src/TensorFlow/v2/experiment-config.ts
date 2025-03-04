// Experiment configuration for tank reinforcement learning
// This allows fine-tuning the RL model and experimenting with different hyperparameters

// Define experiment configurations that can be easily switched
export type RLExperimentConfig = {
    name: string;
    description: string;
    // Network architecture
    hiddenLayers: number[];
    // Learning parameters
    learningRate: number;
    gamma: number;  // Discount factor
    epsilon: number;  // Initial exploration rate
    epsilonMin: number;  // Minimum exploration rate
    epsilonDecay: number;  // Exploration decay rate
    // Memory parameters
    memorySize: number;
    batchSize: number;
    // Training control
    updateTargetEvery: number;  // Update target network every N episodes
    saveModelEvery: number;  // Save model every N episodes
};

// Default experiment configuration
export const DEFAULT_EXPERIMENT: RLExperimentConfig = {
    name: 'default',
    description: 'Balanced configuration for tank RL training',
    // Network architecture
    hiddenLayers: [128, 64],
    // Learning parameters
    learningRate: 0.001,
    gamma: 0.99,
    epsilon: 1.0,
    epsilonMin: 0.15,
    epsilonDecay: 0.9995,
    // Memory parameters
    memorySize: 10000,
    batchSize: 32,
    // Training control
    updateTargetEvery: 10,
    saveModelEvery: 10,
};

// More aggressive experiment - favors shooting and dealing damage
export const AGGRESSIVE_EXPERIMENT: RLExperimentConfig = {
    name: 'aggressive',
    description: 'Configuration favoring offensive actions and dealing damage',
    hiddenLayers: [256, 128, 64],
    learningRate: 0.0015,
    gamma: 0.95,  // More focused on immediate rewards
    epsilon: 1.0,
    epsilonMin: 0.1,
    epsilonDecay: 0.99,  // Slower decay for more exploration
    memorySize: 15000,
    batchSize: 64,
    updateTargetEvery: 8,
    saveModelEvery: 10,
};

// Defensive experiment - favors survival and avoiding damage
export const DEFENSIVE_EXPERIMENT: RLExperimentConfig = {
    name: 'defensive',
    description: 'Configuration favoring survival and damage avoidance',
    hiddenLayers: [128, 128, 64],
    learningRate: 0.0008,
    gamma: 0.99,  // More focus on long-term rewards
    epsilon: 1.0,
    epsilonMin: 0.05,  // Lower min epsilon for better exploitation
    epsilonDecay: 0.998,  // Slower decay
    memorySize: 20000,
    batchSize: 32,
    updateTargetEvery: 12,
    saveModelEvery: 10,
};

// Map of available experiments
export const EXPERIMENTS: { [key: string]: RLExperimentConfig } = {
    default: DEFAULT_EXPERIMENT,
    aggressive: AGGRESSIVE_EXPERIMENT,
    defensive: DEFENSIVE_EXPERIMENT,
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