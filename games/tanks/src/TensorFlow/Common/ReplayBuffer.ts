import { floor, max, min } from '../../../../../lib/math.ts';
import { random } from '../../../../../lib/random.ts';
import { shuffle } from '../../../../../lib/shuffle.ts';

export class ReplayBuffer {
    protected indices: number[];

    constructor(length: number) {
        this.indices = shuffle(Array.from({ length }, (_, i) => i));
    }

    getSample(
        batchSize: number,
        offset: number = 0,
        limit: number = Infinity,
    ) {
        offset = max(offset, 0);
        limit = min(limit, this.indices.length);
        const diff = limit - offset;
        const start = offset + floor(random() * diff);
        const result = [];
        let w = 0;
        let r = start;
        while (result.length < batchSize) {
            result[w++] = this.indices[r++];
            r = r >= limit ? offset : r;
        }

        return result;
    }
}
