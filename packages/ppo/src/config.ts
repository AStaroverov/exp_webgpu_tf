export interface PpoConfig {
    clipNorm: number;
    gamma: (iteration: number) => number;

    /** Fixed entropy bonus coefficient (classic PPO; ~0.01 for discrete). */
    entropyCoeff: number;

    policyEpochs: (iteration: number) => number;
    policyClipRatio: number;
    /**
     * L2 anchor on raw policy logits: `coeff * mean(logits²)` added to the loss.
     * Softmax is shift-invariant and saturated actions get ~zero entropy
     * gradient, so without an anchor logits drift unboundedly (hundreds). Keeps
     * the scale near zero without changing relative preferences. 0/undefined = off.
     */
    policyLogitsL2?: number;

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
