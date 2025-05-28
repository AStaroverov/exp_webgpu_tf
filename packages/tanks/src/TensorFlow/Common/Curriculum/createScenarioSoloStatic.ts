import { createBattlefield } from './createBattlefield.ts';
import { Scenario } from './types.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';
import { createScenarioBase } from './createScenarioBase.ts';

export const indexScenarioSoloStatic = 0;

export async function createScenarioSoloStatic(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const scenario = await createScenarioBase({
        ...options,
        size: randomRangeInt(800, 1000),
        alliesCount: 1,
        enemiesCount: randomRangeInt(1, 3),
    });
    scenario.index = indexScenarioSoloStatic;
    return scenario;
}