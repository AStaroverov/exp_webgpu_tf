import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { createScenarioWithAlliesActive } from './createScenarioWithAlliesActive.ts';
import { fillWithRandomHistoricalAgents } from './Utils/fillWithRandomHistoricalAgents.ts';

export const indexScenarioWithHistoricalAgents = 7;

export async function createScenarioWithHistoricalAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioWithAlliesActive(options);
    episode.index = indexScenarioWithHistoricalAgents;

    fillWithRandomHistoricalAgents(episode);

    return episode;
}

