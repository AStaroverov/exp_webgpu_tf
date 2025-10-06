import { clamp } from 'lodash-es';
import { CONFIG } from './config.ts';

export function getDynamicLearningRate(kl: number, lr: number): number {
    const {
        klConfig: { high: klHigh, low: klLow },
        lrConfig: { multHigh: lrMultHigh, multLow: lrMultLow, min: minLR, max: maxLR },
    } = CONFIG;

    if (kl > klHigh) lr *= lrMultHigh;
    else if (kl < klLow) lr *= lrMultLow;

    return clamp(lr, minLR, maxLR);
}