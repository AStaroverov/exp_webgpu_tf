import { createScenarioAgentsVsBots } from './createScenarioAgentsVsBots.ts';
import { createScenarioGridBase } from './createScenarioGridBase.ts';
import { Scenario } from './types.ts';

export function createScenarioAgentsVsBots1(options: Parameters<typeof createScenarioGridBase>[0]): Scenario {
    const episode = createScenarioAgentsVsBots(1, options);
    return episode;
}

