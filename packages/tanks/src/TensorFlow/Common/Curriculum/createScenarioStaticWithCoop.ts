import { createScenarioWithAlliesStatic } from './createScenarioWithAlliesStatic.ts';
import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { fillAlliesWithActorAgents } from './Utils/fillAlliesWithActorAgents.ts';

export const indexScenarioStaticWithCoop = 2;

export async function createScenarioStaticWithCoop(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioWithAlliesStatic(options);
    episode.index = indexScenarioStaticWithCoop;

    const destroy = await fillAlliesWithActorAgents(episode);

    return {
        ...episode,
        destroy: () => {
            episode.destroy();
            destroy();
        },
    };
}