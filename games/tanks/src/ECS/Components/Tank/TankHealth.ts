import { min, smoothstep } from '../../../../../../lib/math.ts';
import { Tank } from './Tank.ts';
import { getTankCurrentPartsCount } from './TankUtils.ts';

export const HEALTH_THRESHOLD = 0.75;

// return from 0 to 1
export function getTankHealth(tankEid: number): number {
    const initialPartsCount = Tank.initialPartsCount[tankEid];
    const partsCount = getTankCurrentPartsCount(tankEid);
    const absHealth = min(1, partsCount / initialPartsCount);
    const health = smoothstep(HEALTH_THRESHOLD, 1, absHealth);

    return health;
}