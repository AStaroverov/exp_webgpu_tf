import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { createScenarioAgentsVsBots } from './createScenarioAgentsVsBots.ts';

export const indexScenarioAgentsVsBots1 = 2;

export async function createScenarioAgentsVsBots1(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioAgentsVsBots(1, options);
    episode.index = indexScenarioAgentsVsBots1;
    return episode;
}

