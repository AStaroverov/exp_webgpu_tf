import { clamp } from 'lodash-es';
import { CONFIG } from './config.ts';

export function getDynamicLearningRate(kl: number, lr: number): number {
    const {
        lrConfig: { kl: { high: klHigh, low: klLow }, multHigh: lrMultHigh, multLow: lrMultLow, min: minLR, max: maxLR },
    } = CONFIG;

    if (kl > klHigh) lr *= lrMultHigh;
    else if (kl < klLow) lr *= lrMultLow;

    return clamp(lr, minLR, maxLR);
}

export function getDynamicPerturb(klNoise: number, scale: number): number {
    const {
        perturbWeightsConfig: { kl: { high: klHigh, low: klLow }, multHigh, multLow, min, max },
    } = CONFIG;

    if (klNoise > klHigh) scale *= multHigh;
    else if (klNoise < klLow) scale *= multLow;

    return clamp(scale, min, max);
}