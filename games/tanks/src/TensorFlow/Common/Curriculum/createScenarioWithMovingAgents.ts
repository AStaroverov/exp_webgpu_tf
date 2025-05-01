import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { randomRangeFloat } from '../../../../../../lib/random.ts';
import { createScenarioStaticWithCoop } from './createScenarioStaticWithCoop.ts';
import { getScenarioIndex } from './utils.ts';
import { addSimpleHeuristicAgents } from './addSimpleHeuristicAgents.ts';

export const indexScenarioWithMovingAgents = getScenarioIndex();

export async function createScenarioWithMovingAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioStaticWithCoop(options);
    episode.index = indexScenarioWithMovingAgents;

    addSimpleHeuristicAgents(episode, {
        move: randomRangeFloat(0.4, 1),
    });

    return episode;
}

