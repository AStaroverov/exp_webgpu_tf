import { CONFIG } from '../PPO/config.ts';

export function getDynamicLearningRate(kl: number, lr: number): number {
    const {
        klConfig: { target: klTarget, highCoef: klHighCoef, lowCoef: klLowCoef },
        lrConfig: { multHigh: lrMultHigh, multLow: lrMultLow, min: minLR, max: maxLR },
    } = CONFIG;

    // Если KL слишком большая (в X раз больше, чем target)
    if (kl > klHighCoef * klTarget) {
        lr = Math.max(lr * lrMultHigh, minLR);
    }
    // Если KL слишком маленькая
    else if (kl < klLowCoef * klTarget) {
        lr = Math.min(lr * lrMultLow, maxLR);
    }

    return lr;
}