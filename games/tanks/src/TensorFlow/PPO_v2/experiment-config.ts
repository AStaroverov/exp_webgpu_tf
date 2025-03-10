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
    // PPO-specific parameters
    ppoEpsilon: number;             // PPO clipping parameter
    ppoEpochs: number;              // Number of epochs to train on the same data
    entropyCoeff: number;           // Entropy coefficient for encouraging exploration
    // Training control
    saveModelEvery: number;         // Save model every N episodes
};

// Default experiment configuration for PPO
export const DEFAULT_EXPERIMENT: RLExperimentConfig = {
    name: 'ppo-default',
    description: 'Balanced PPO configuration for tank RL training',
    // Network architecture
    hiddenLayers: [256, 256, 256, 128],
    // Learning parameters
    learningRate: 0.0005,           // Generally smaller for PPO
    gamma: 0.99,
    lam: 0.95,                      // GAE lambda for advantage estimation
    // PPO-specific parameters
    ppoEpsilon: 0.15,                // Standard PPO clipping parameter
    ppoEpochs: 4,                   // Train 4 epochs over the same data
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