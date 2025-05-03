import { RingBuffer } from 'ring-buffer-ts';
import { max, min } from '../../../../../../lib/math.ts';
import { clamp } from 'lodash-es';

export function createLossChecker() {
    const history = new RingBuffer<number>(100);

    return {
        check: (losses: number[]) => {
            if (history.getBufferLength() < 10) return true;

            const arr = history.toArray();
            const minLoss = min(...arr);
            const maxLoss = max(...arr);
            const diff = maxLoss - minLoss;
            const shift = diff * 500;

            const result = losses.every(loss => loss === clamp(loss, minLoss - shift, maxLoss + shift));

            history.add(...losses);

            return result;
        },
    };
}