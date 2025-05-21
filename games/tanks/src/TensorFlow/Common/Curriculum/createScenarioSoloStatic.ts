import { createBattlefield } from './createBattlefield.ts';
import { Scenario } from './types.ts';
import { createScenarioWithAlliesStatic } from './createScenarioWithAlliesStatic.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';

export const indexScenarioSoloStatic = 0;

export async function createScenarioSoloStatic(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const scenario = await createScenarioWithAlliesStatic({
        ...options,
        size: randomRangeInt(300, 700),
        alliesCount: 1,
        enemiesCount: randomRangeInt(1, 3),
    });
    scenario.index = indexScenarioSoloStatic;
    return scenario;
}