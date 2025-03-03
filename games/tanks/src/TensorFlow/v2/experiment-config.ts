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
    // Reward weighting
    rewardWeights: {
        healthChange: number;
        healthBonus: number;
        aimQuality: number;
        shootingAimed: number;
        shootingRandom: number;
        bulletAvoidance: number;
        movementBase: number;
        strategicMovement: number;
        survival: number;
        mapBorder: number;
        borderGradient: number;
        distanceKeeping: number;
        victory: number;
        death: number;
    };
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
    epsilonMin: 0.1,
    epsilonDecay: 0.995,
    // Memory parameters
    memorySize: 10000,
    batchSize: 32,
    // Training control
    updateTargetEvery: 10,
    saveModelEvery: 10,
    // Reward weighting - default weights from calculateReward_V2.ts
    rewardWeights: {
        healthChange: 0.5,
        healthBonus: 0.05,
        aimQuality: 2.5,
        shootingAimed: 1.0,
        shootingRandom: -0.5,
        bulletAvoidance: -2.0,
        movementBase: 0.05,
        strategicMovement: 0.2,
        survival: 0.02,
        mapBorder: -3.0,
        borderGradient: -0.5,
        distanceKeeping: 0.3,
        victory: 10.0,
        death: -5.0,
    },
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
    rewardWeights: {
        healthChange: 0.3,        // Less penalty for taking damage
        healthBonus: 0.02,
        aimQuality: 4.0,          // Much higher reward for good aiming
        shootingAimed: 2.0,       // Double reward for shooting when aimed
        shootingRandom: -0.2,     // Less penalty for random shooting
        bulletAvoidance: -1.5,    // Less penalty for not avoiding bullets
        movementBase: 0.03,
        strategicMovement: 0.1,
        survival: 0.01,           // Less survival bonus
        mapBorder: -3.0,
        borderGradient: -0.5,
        distanceKeeping: 0.15,    // Less reward for keeping distance
        victory: 12.0,            // Higher victory reward
        death: -4.0,               // Less death penalty
    },
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
    rewardWeights: {
        healthChange: 0.8,         // Higher penalty for taking damage
        healthBonus: 0.1,          // Higher reward for maintaining health
        aimQuality: 1.5,           // Lower reward for aiming
        shootingAimed: 0.8,        // Lower reward for shooting
        shootingRandom: -1.0,      // Higher penalty for random shooting
        bulletAvoidance: -3.0,     // Higher penalty for not avoiding bullets
        movementBase: 0.1,         // Higher reward for movement
        strategicMovement: 0.4,    // Higher reward for strategic movement
        survival: 0.05,            // Higher survival bonus
        mapBorder: -4.0,           // Higher penalty for border
        borderGradient: -0.8,      // Higher gradient penalty
        distanceKeeping: 0.5,      // Higher reward for optimal distance
        victory: 8.0,              // Lower victory reward
        death: -8.0,                // Higher death penalty
    },
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