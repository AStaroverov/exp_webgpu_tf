import { abs } from '../../../../../../lib/math.ts';

export function isLossDangerous(loss: number): boolean {
    return !Number.isFinite(loss) || abs(loss) > 50;
}