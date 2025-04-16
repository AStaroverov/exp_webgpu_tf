import { random, randomRangeInt } from '../../../../../lib/random.ts';
import { abs, floor, max } from '../../../../../lib/math.ts';
import { ReplayBuffer } from './ReplayBuffer.ts';

export class PrioritizedReplayBuffer extends ReplayBuffer {
    // public alpha: number;
    // public beta: number;
    // private probs: Float32Array;

    private sortedIndices: number[];

    constructor(tdErrors: Float32Array) {
        super(tdErrors.length);

        // this.alpha = alpha ?? computeAlpha(tdErrors);
        // const priorities = tdErrors.map(tdError => pow(abs(tdError) + 1e-6, this.alpha));
        // const totalPriority = priorities.reduce((a, b) => a + b, 0);
        // this.probs = priorities.map(p => p / totalPriority);
        // this.beta = beta ?? computeBeta(this.probs);

        const absTdErrors = tdErrors.map(abs);
        this.sortedIndices = this.indices.slice().sort((a, b) => absTdErrors[b] - absTdErrors[a]);
    }

    getSampleWithTop(
        batchSize: number,
        offset: number = 0,
        limit: number = Infinity,
        topPercent: number = 0.2,
        topPickChance: number = 0.3,
    ) {
        const len = this.sortedIndices.length;
        const topLen = max(1, floor(len * topPercent));
        const indices = super.getSample(batchSize, offset, limit);

        for (let i = 0; i < indices.length; i++) {
            if (random() < topPickChance) {
                indices[i] = this.sortedIndices[randomRangeInt(0, topLen - 1)];
            }
        }

        return indices;
    }

    //
    // getWeights(indices: number[]) {
    //     const len = this.probs.length;
    //     const weights = indices.map(i => pow(len * this.probs[i], -this.beta));
    //     const maxWeight = max(...weights);
    //     return weights.map(w => w / maxWeight);
    // }
}

//
// function computeAlpha(tdErrors: Float32Array, mi = 0.3, ma = 0.6): number {
//     const absTdErrors = tdErrors.map(abs);
//     const m = mean(absTdErrors);
//     const s = std(absTdErrors, m);
//     const score = min(s / (m + 1e-6), 1);
//     return mi + (ma - mi) * score;
// }
//
// function computeBeta(probs: Float32Array, min = 0.4, max = 1.0): number {
//     const entropy = -probs.reduce((acc, p) => acc + (p > 0 ? p * log(p) : 0), 0);
//     const uniformEntropy = log(probs.length);
//     const imbalance = 1 - entropy / uniformEntropy;
//     return min + (max - min) * imbalance;
// }
