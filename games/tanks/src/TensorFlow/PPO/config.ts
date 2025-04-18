// Experiment configuration for tank reinforcement learning with PPO
// This allows fine-tuning the RL model and experimenting with different hyperparameters

// Define experiment configurations that can be easily switched

export type Config = {
    name: string;
    // Learning parameters
    clipNorm: number;
    gamma: number;                  // Discount factor
    lam: number;                    // GAE lambda
    // PPO-specific parameters
    epochs: number;                // Number of epochs to train on the same data
    clipRatio: number;             // Clipping ratio for PPO
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

    batchSize: number;              // Batch size for worker
    miniBatchSize: number,
    warmupFrames: number;              // Maximum number of frames to train on
    episodeFrames: number;              // Maximum number of frames to train on
    // Workers
    workerCount: number;                // Number of parallel workers
    // Training control
    savePath: string;
    fsModelPath?: string;
};

// Default experiment configuration for PPO
export const DEFAULT_EXPERIMENT: Config = {
    name: 'ppo-default',
    // Learning parameters
    clipNorm: 5,
    gamma: 0.99,
    lam: 0.95,
    // PPO-specific parameters
    epochs: 5,
    clipRatio: 0.2,
    entropyCoeff: 0.01,

    klConfig: {
        target: 0.02,
        highCoef: 1.5,       // Если KL > 2 * 0.01 => 0.02
        lowCoef: 0.5,    // Если KL < 0.5 * 0.01 => 0.005
        max: 0.5,
    },
    lrConfig: {
        initial: 1e-4,
        multHigh: 0.9,
        multLow: 1.1,
        min: 1e-6,
        max: 5e-3,
    },

    batchSize: 256, // useless for appo
    miniBatchSize: 64,
    // Training parameters
    warmupFrames: 100,
    episodeFrames: 1200, // usually produce 250 samples
    // Workers
    workerCount: 8,
    // Training control
    savePath: 'APPO_v1',
    fsModelPath: 'v6',
};


// Current active experiment
export let CONFIG: Config = DEFAULT_EXPERIMENT;
