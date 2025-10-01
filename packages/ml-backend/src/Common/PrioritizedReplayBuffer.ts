import { abs, floor, max } from '../../../../lib/math.ts';
import { random, randomRangeInt } from '../../../../lib/random.ts';
import { ReplayBuffer } from './ReplayBuffer.ts';

export class PrioritizedReplayBuffer extends ReplayBuffer {
    private sortedIndices: number[];

    constructor(tdErrors: Float32Array) {
        super(tdErrors.length);

        const absTdErrors = tdErrors.map(abs);
        this.sortedIndices = this.indices.slice().sort((a, b) => absTdErrors[b] - absTdErrors[a]);
    }

    getSampleWithTop(
        batchSize: number,
        offset: number = 0,
        limit: number = Infinity,
        topPercent: number = 0.2,
        topPickChance: number = 0.2,
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
}
