import { createScenarioAgentsVsBots } from './createScenarioAgentsVsBots.ts';
import { createScenarioBase } from './createScenarioBase.ts';
import { Scenario } from './types.ts';

export function createScenarioAgentsVsBots1(options: Parameters<typeof createScenarioBase>[0]): Scenario {
    const episode = createScenarioAgentsVsBots(1, options);
    return episode;
}

