import { createBattlefield } from './createBattlefield.ts';
import { createScenarioAgentsVsBots } from './createScenarioAgentsVsBots.ts';
import { Scenario } from './types.ts';

export const indexScenarioAgentsVsBots2 = 2;

export function createScenarioAgentsVsBots2(options: Parameters<typeof createBattlefield>[0]): Scenario {
    const episode = createScenarioAgentsVsBots(2, options);
    episode.index = indexScenarioAgentsVsBots2;
    return episode;
}

