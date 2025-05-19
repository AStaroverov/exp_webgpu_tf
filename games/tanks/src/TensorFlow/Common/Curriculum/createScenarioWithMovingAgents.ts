import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { randomRangeFloat } from '../../../../../../lib/random.ts';
import { createScenarioStaticWithCoop } from './createScenarioStaticWithCoop.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';

export const indexScenarioWithMovingAgents = 2;

export async function createScenarioWithMovingAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioStaticWithCoop(options);
    episode.index = indexScenarioWithMovingAgents;

    fillWithSimpleHeuristicAgents(episode, {
        move: randomRangeFloat(0.4, 1),
    });

    return episode;
}

