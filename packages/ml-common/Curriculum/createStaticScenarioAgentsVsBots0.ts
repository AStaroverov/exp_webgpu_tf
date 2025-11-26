import { createScenarioAgentsVsBots } from './createScenarioAgentsVsBots.ts';
import { createScenarioBase } from './createScenarioBase.ts';
import { Scenario } from './types.ts';

export const indexStaticScenarioAgentsVsBots0 = 0;

export function createStaticScenarioAgentsVsBots0(options: Parameters<typeof createScenarioBase>[0]): Scenario {
    const scenario = createScenarioAgentsVsBots(0, {
        ...options,
        alliesCount: 3,
        enemiesCount: 3,
    });
    scenario.index = indexStaticScenarioAgentsVsBots0;
    return scenario;
}
