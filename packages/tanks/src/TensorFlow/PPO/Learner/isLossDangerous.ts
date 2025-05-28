import { abs } from '../../../../../../lib/math.ts';

export function isLossDangerous(loss: number, threshold: number): boolean {
    return !Number.isFinite(loss) || abs(loss) > threshold;
}