import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { createScenarioStaticWithCoop } from './createScenarioStaticWithCoop.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';

export const indexScenarioWithStrongHeuristicAgents = 5;

export async function createScenarioWithStrongHeuristicAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioStaticWithCoop(options);
    episode.index = indexScenarioWithStrongHeuristicAgents;

    fillWithSimpleHeuristicAgents(episode, {
        move: 1,
        aim: {
            aimError: 0.2,
            shootChance: 0.8,
        },
    });

    return episode;
}

