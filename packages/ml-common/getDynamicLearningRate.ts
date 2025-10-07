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

export function getDynamicPerturb(klPure: number, klNoise: number, scale: number): number {
    const {
        lrConfig,
        perturbWeightsConfig: { kl: { high: klHigh, low: klLow }, multHigh, multLow, min, max },
    } = CONFIG;

    if (klPure > lrConfig.kl.high) {
        return 0; // Disable perturbations entirely
    };

    if (klNoise > klHigh) scale *= multHigh;
    else if (klNoise < klLow) scale *= multLow;

    return clamp(scale, min, max);
}