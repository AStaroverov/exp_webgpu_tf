import { CONFIG } from '../PPO/Common/config.ts';

export function getConfigPatch(
    kl: number,
    lr: number,
    clip: number,
): { newLR: number; newClip: number } {
    const {
        klConfig: { target: klTarget, highCoef: klHighCoef, lowCoef: klLowCoef },
        lrConfig: { multHigh: lrMultHigh, multLow: lrMultLow, min: minLR, max: maxLR },
        clipRatioConfig: { deltaHigh: clipDeltaHigh, deltaLow: clipDeltaLow, min: minClip, max: maxClip },
    } = CONFIG;

    let newLR = lr;
    let newClip = clip;

    // Если KL слишком большая (в X раз больше, чем target)
    if (kl > klHighCoef * klTarget) {
        newLR = Math.max(lr * lrMultHigh, minLR);
        newClip = Math.max(clip - clipDeltaHigh, minClip);
    }
    // Если KL слишком маленькая
    else if (kl < klLowCoef * klTarget) {
        newLR = Math.min(lr * lrMultLow, maxLR);
        newClip = Math.min(clip + clipDeltaLow, maxClip);
    }

    return { newLR, newClip };
}