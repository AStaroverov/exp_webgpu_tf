import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { createScenarioAgentsVsBots } from './createScenarioAgentsVsBots.ts';

export const indexScenarioAgentsVsBots2 = 3;

export async function createScenarioAgentsVsBots2(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioAgentsVsBots(2, options);
    episode.index = indexScenarioAgentsVsBots2;
    return episode;
}

