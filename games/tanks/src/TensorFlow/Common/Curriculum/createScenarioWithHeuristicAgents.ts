import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { randomRangeFloat } from '../../../../../../lib/random.ts';
import { createScenarioStaticWithCoop } from './createScenarioStaticWithCoop.ts';
import { getScenarioIndex } from './utils.ts';
import { addSimpleHeuristicAgents } from './addSimpleHeuristicAgents.ts';

export const indexScenarioWithHeuristicAgents = getScenarioIndex();

export async function createScenarioWithHeuristicAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioStaticWithCoop(options);
    episode.index = indexScenarioWithHeuristicAgents;

    addSimpleHeuristicAgents(episode, {
        move: randomRangeFloat(0.8, 1),
        aim: {
            aimError: randomRangeFloat(1, 3),
            shootChance: randomRangeFloat(0.2, 0.4),
        },
    });

    return episode;
}

