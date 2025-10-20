import { abs } from '../../../../../lib/math.ts';

export function isLossDangerous(loss: number, threshold: number = 1000): boolean {
    return !Number.isFinite(loss) || abs(loss) > threshold;
}
