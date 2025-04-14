import { random } from '../../../../../lib/random.ts';

export class PrioritizedReplayBuffer {
    private cdf: Float32Array;
    private probs: Float32Array;

    constructor(tdErrors: Float32Array, alpha = 0.6) {
        const priorities = tdErrors.map(tdError => Math.pow(Math.abs(tdError) + 1e-6, alpha));
        const totalPriority = priorities.reduce((a, b) => a + b, 0);
        this.probs = priorities.map(p => p / totalPriority);
        this.cdf = createCDF(this.probs);
        console.log('>> a', computeAlpha(tdErrors).toFixed(4), 0.6);
        console.log('>> b', computeBeta(this.probs).toFixed(4), 0.4);
    }

    sample(batchSize: number, beta = 0.4) {

        const { cdf, probs } = this;
        const result = [];

        while (result.length < batchSize) {
            const rand = random() * cdf[cdf.length - 1];
            const index = binarySearchCDF(cdf, rand);
            result.push(index);
        }

        // Importance sampling weights
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
        const mid = Math.floor((left + right) / 2);
        if (r <= cdf[mid]) {
            right = mid; // сдвигаемся влево
        } else {
            left = mid + 1; // сдвигаемся вправо
        }
    }

    return left; // минимальный i, где cdf[i] ≥ r
}

function computeAlpha(tdErrors: Float32Array, min = 0.3, max = 0.9): number {
    const mean = tdErrors.reduce((a, b) => a + Math.abs(b), 0) / tdErrors.length;
    const variance = tdErrors.reduce((acc, e) => acc + Math.pow(Math.abs(e) - mean, 2), 0) / tdErrors.length;
    const std = Math.sqrt(variance);

    const score = Math.min(std / (mean + 1e-6), 1); // насколько ошибки разнообразны
    return min + (max - min) * score;
}

function computeBeta(probs: Float32Array, min = 0.4, max = 1.0): number {
    const entropy = -probs.reduce((acc, p) => acc + (p > 0 ? p * Math.log(p) : 0), 0);
    const uniformEntropy = Math.log(probs.length); // максимум при равномерности
    const imbalance = 1 - entropy / uniformEntropy; // 0 = uniform, 1 = max bias

    return min + (max - min) * imbalance;
}
