import { createBattlefield } from './createBattlefield.ts';
import { createScenarioAgentsVsBots } from './createScenarioAgentsVsBots.ts';
import { Scenario } from './types.ts';

export const indexScenarioAgentsVsBots1 = 2;

export function createScenarioAgentsVsBots1(options: Parameters<typeof createBattlefield>[0]): Scenario {
    const episode = createScenarioAgentsVsBots(1, options);
    episode.index = indexScenarioAgentsVsBots1;
    return episode;
}

