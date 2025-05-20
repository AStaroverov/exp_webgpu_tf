import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { createScenarioStaticWithCoop } from './createScenarioStaticWithCoop.ts';
import { fillWithRandomHistoricalAgents } from './Utils/fillWithRandomHistoricalAgents.ts';

export const indexScenarioWithHistoricalAgents = 7;

export async function createScenarioWithHistoricalAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioStaticWithCoop(options);
    episode.index = indexScenarioWithHistoricalAgents;

    const destroy = await fillWithRandomHistoricalAgents(episode);

    return {
        ...episode,
        destroy: () => {
            destroy();
            episode.destroy();
        },
    };
}

