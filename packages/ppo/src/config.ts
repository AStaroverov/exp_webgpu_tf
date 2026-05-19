export interface PpoConfig {
    clipNorm: number;
    gamma: (iteration: number) => number;

    adaptiveEntropy: {
        targetRatio: number;
        alphaLR: number;
        initialLogAlpha: number;
        minLogAlpha: number;
        maxLogAlpha: number;
    };

    policyEpochs: (iteration: number) => number;
    policyClipRatio: number;

    valueEpochs: (iteration: number) => number;
    valueClipRatio: number;
    valueLossCoeff: number;
    valueLRCoeff: number;

    lrConfig: {
        kl: { high: number; target: number; low: number };
        initial: number;
        multHigh: number;
        multLow: number;
        min: number;
        max: number;
    };

    batchSize: (iteration: number) => number;
    miniBatchSize: (iteration: number) => number;

    backpressureQueueSize: number;
    savePath: string;
}
