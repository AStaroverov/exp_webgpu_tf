import { random } from '../../../../../lib/random.ts';
import { abs, floor, log, mean, min, std } from '../../../../../lib/math.ts';

export class PrioritizedReplayBuffer {
    public alpha: number;
    public beta: number;

    private cdf: Float32Array;
    private probs: Float32Array;

    constructor(tdErrors: Float32Array, alpha?: number, beta?: number) {
        this.alpha = alpha ?? computeAlpha(tdErrors);
        const priorities = tdErrors.map(tdError => Math.pow(Math.abs(tdError) + 1e-6, this.alpha));
        const totalPriority = priorities.reduce((a, b) => a + b, 0);
        this.probs = priorities.map(p => p / totalPriority);
        this.beta = beta ?? computeBeta(this.probs);
        this.cdf = createCDF(this.probs);
    }

    sample(batchSize: number) {
        const { cdf, beta, probs } = this;
        const result = [];

        while (result.length < batchSize) {
            const rand = random() * cdf[cdf.length - 1];
            const index = binarySearchCDF(cdf, rand);
            result.push(index);
        }

        const N = cdf.length;
        const weights = result.map(i => Math.pow(N * probs[i], -beta));
        const maxWeight = Math.max(...weights);
        const normalizedWeights = weights.map(w => w / maxWeight);

        return { indices: result, weights: normalizedWeights };
    }
}

function createCDF(probs: Float32Array) {
    const cdf = new Float32Array(probs.length);
    let accum = 0;
    for (let i = 0; i < probs.length; i++) {
        accum += probs[i];
        cdf[i] = accum;
    }
    cdf[cdf.length - 1] = 1;
    return cdf;
}

function binarySearchCDF(cdf: Float32Array, r: number): number {
    let left = 0;
    let right = cdf.length - 1;

    while (left < right) {
        const mid = floor((left + right) / 2);
        if (r <= cdf[mid]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    return left;
}

function computeAlpha(tdErrors: Float32Array, mi = 0.3, ma = 0.6): number {
    const absTdErrors = tdErrors.map(abs);
    const m = mean(absTdErrors);
    const s = std(absTdErrors, m);
    const score = min(s / (m + 1e-6), 1);
    return mi + (ma - mi) * score;
}

function computeBeta(probs: Float32Array, min = 0.4, max = 1.0): number {
    const entropy = -probs.reduce((acc, p) => acc + (p > 0 ? p * log(p) : 0), 0);
    const uniformEntropy = log(probs.length);
    const imbalance = 1 - entropy / uniformEntropy;
    return min + (max - min) * imbalance;
}
