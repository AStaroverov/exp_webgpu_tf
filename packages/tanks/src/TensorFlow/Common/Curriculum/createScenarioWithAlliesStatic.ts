import { randomRangeInt } from '../../../../../../lib/random.ts';
import { Scenario } from './types.ts';
import { createScenarioBase } from './createScenarioBase.ts';

export const indexScenarioWithAlliesStatic = 1;

export async function createScenarioWithAlliesStatic(options: Parameters<typeof createScenarioBase>[0]): Promise<Scenario> {
    const scenario = await createScenarioBase({
        ...options,
        size: randomRangeInt(1000, 1600),
        alliesCount: randomRangeInt(1, 3),
        enemiesCount: randomRangeInt(1, 3),
    });
    scenario.index = indexScenarioWithAlliesStatic;
    return scenario;
}