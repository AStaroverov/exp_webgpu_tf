import { createBattlefield } from './createBattlefield.ts';
import { Scenario } from './types.ts';
import { createScenarioWithAlliesStatic } from './createScenarioWithAlliesStatic.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';

export const indexScenarioSoloStatic = 0;

export async function createScenarioSoloStatic(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    return createScenarioWithAlliesStatic({
        ...options,
        alliesCount: 1,
        enemiesCount: randomRangeInt(1, 3),
    });
}