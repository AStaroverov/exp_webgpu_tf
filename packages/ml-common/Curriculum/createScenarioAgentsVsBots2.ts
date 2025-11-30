import { createScenarioAgentsVsBots } from './createScenarioAgentsVsBots.ts';
import { createScenarioBase } from './createScenarioBase.ts';
import { Scenario } from './types.ts';

export function createScenarioAgentsVsBots2(options: Parameters<typeof createScenarioBase>[0]): Scenario {
    const episode = createScenarioAgentsVsBots(2, options);
    return episode;
}

