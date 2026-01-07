import { createScenarioAgentsVsBots } from './createScenarioAgentsVsBots.ts';
import { createScenarioGridBase } from './createScenarioGridBase.ts';
import { Scenario } from './types.ts';

export function createStaticScenarioAgentsVsBots0(options: Parameters<typeof createScenarioGridBase>[0]): Scenario {
    const scenario = createScenarioAgentsVsBots(0, {
        ...options,
        alliesCount: 3,
        enemiesCount: 3,
    });
    return scenario;
}
