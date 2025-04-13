import { random } from '../../../../../lib/random.ts';
import { binarySearch } from '../../../../../lib/binarySearch.ts';

export class PrioritizedReplayBuffer {
    private cdf: Float32Array;
    private probs: Float32Array;

    constructor(tdErrors: Float32Array, alpha = 0.6) {
        const priorities = tdErrors.map(tdError => Math.pow(Math.abs(tdError) + 1e-6, alpha));
        const totalPriority = priorities.reduce((a, b) => a + b, 0);
        this.probs = priorities.map(p => p / totalPriority);
        this.cdf = createCDF(this.probs);
    }

    sample(batchSize: number, beta = 0.4) {
        const { cdf, probs } = this;
        const result = [];

        while (result.length < batchSize) {
            const rand = random() * cdf[cdf.length - 1];
            const index = binarySearch(0, cdf.length - 1, i => cdf[i] - rand);
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