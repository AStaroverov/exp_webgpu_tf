import { RingBuffer } from 'ring-buffer-ts';
import { abs, mean } from '../../../../../../lib/math.ts';
import { clamp } from 'lodash-es';

export function createLossChecker() {
    let lastMean = 0;
    const history = new RingBuffer<number>(100);

    return {
        check: (losses: number[]) => {
            if (history.getBufferLength() < 10) return true;

            const shift = abs(lastMean) * 500;
            const result = losses.every(loss => loss === clamp(loss, lastMean - shift, lastMean + shift));

            history.add(...losses);
            lastMean = mean(history.toArray());

            return result;
        },
    };
}