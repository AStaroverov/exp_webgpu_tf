import { CONFIG } from './config.ts';

export function getDynamicLearningRate(kl: number, lr: number): number {
    const {
        klConfig: { high: klHigh, low: klLow },
        lrConfig: { multHigh: lrMultHigh, multLow: lrMultLow, min: minLR, max: maxLR },
    } = CONFIG;

    if (kl > klHigh) {
        lr = Math.max(lr * lrMultHigh, minLR);
    }
    // Если KL слишком маленькая
    else if (kl < klLow) {
        lr = Math.min(lr * lrMultLow, maxLR);
    }

    return lr;
}