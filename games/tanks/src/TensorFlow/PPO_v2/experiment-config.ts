// Experiment configuration for tank reinforcement learning with PPO
// This allows fine-tuning the RL model and experimenting with different hyperparameters

// Define experiment configurations that can be easily switched
export type RLExperimentConfig = {
    name: string;
    description: string;
    // Network architecture
    hiddenLayers: number[];
    // Learning parameters
    learningRate: number;
    gamma: number;                  // Discount factor
    lam: number;                    // GAE lambda
    epsilon: number;                // Initial exploration rate
    epsilonMin: number;             // Minimum exploration rate
    epsilonDecay: number;           // Exploration decay rate
    // PPO-specific parameters
    ppoEpsilon: number;             // PPO clipping parameter
    ppoEpochs: number;              // Number of epochs to train on the same data
    entropyCoeff: number;           // Entropy coefficient for encouraging exploration
    // Memory parameters @deprecated
    batchSize: number;
    // Training control
    saveModelEvery: number;         // Save model every N episodes
};

// Default experiment configuration for PPO
export const DEFAULT_EXPERIMENT: RLExperimentConfig = {
    name: 'ppo-default',
    description: 'Balanced PPO configuration for tank RL training',
    // Network architecture
    hiddenLayers: [128, 64],
    // Learning parameters
    learningRate: 0.0003,           // Generally smaller for PPO
    gamma: 0.99,
    lam: 0.95,                      // GAE lambda for advantage estimation
    epsilon: 1.0,
    epsilonMin: 0.15,
    epsilonDecay: 0.9995,
    // PPO-specific parameters
    ppoEpsilon: 0.2,                // Standard PPO clipping parameter
    ppoEpochs: 4,                   // Train 4 epochs over the same data
    entropyCoeff: 0.01,             // Entropy regularization coefficient
    // Memory parameters
    batchSize: 128,                 // Larger batch for stable PPO training
    // Training control
    saveModelEvery: 10,
};

// More aggressive experiment - favors shooting and dealing damage
export const AGGRESSIVE_EXPERIMENT: RLExperimentConfig = {
    name: 'ppo-aggressive',
    description: 'PPO configuration favoring offensive actions and dealing damage',
    hiddenLayers: [256, 128, 64],
    learningRate: 0.0004,
    gamma: 0.95,                    // More focused on immediate rewards
    lam: 0.9,                       // Lower lambda for more focus on immediate rewards
    epsilon: 1.0,
    epsilonMin: 0.1,
    epsilonDecay: 0.99,             // Slower decay for more exploration
    ppoEpsilon: 0.3,                // Higher clipping allows more policy updates
    ppoEpochs: 6,                   // More training iterations per batch
    entropyCoeff: 0.02,             // Higher entropy for more exploration
    batchSize: 192,
    saveModelEvery: 10,
};

// Defensive experiment - favors survival and avoiding damage
export const DEFENSIVE_EXPERIMENT: RLExperimentConfig = {
    name: 'ppo-defensive',
    description: 'PPO configuration favoring survival and damage avoidance',
    hiddenLayers: [128, 128, 64],
    learningRate: 0.0002,
    gamma: 0.99,                    // More focus on long-term rewards
    lam: 0.95,                      // Standard lambda for balanced advantage estimation
    epsilon: 1.0,
    epsilonMin: 0.05,               // Lower min epsilon for better exploitation
    epsilonDecay: 0.998,            // Slower decay
    ppoEpsilon: 0.15,               // More conservative updates
    ppoEpochs: 3,                   // Fewer epochs to prevent overfitting
    entropyCoeff: 0.005,            // Lower entropy for more conservative behavior
    batchSize: 128,
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