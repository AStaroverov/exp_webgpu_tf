import { createScenarioAgentsVsBots } from './createScenarioAgentsVsBots.ts';
import { createScenarioGridBase } from './createScenarioGridBase.ts';
import { Scenario } from './types.ts';

export function createScenarioAgentsVsBots2(options: Parameters<typeof createScenarioGridBase>[0]): Scenario {
    const episode = createScenarioAgentsVsBots(2, options);
    return episode;
}

