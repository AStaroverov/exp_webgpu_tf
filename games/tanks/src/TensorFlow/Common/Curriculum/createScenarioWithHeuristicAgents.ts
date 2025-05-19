import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { randomRangeFloat } from '../../../../../../lib/random.ts';
import { createScenarioStaticWithCoop } from './createScenarioStaticWithCoop.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';

export const indexScenarioWithHeuristicAgents = 4;

export async function createScenarioWithHeuristicAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioStaticWithCoop(options);
    episode.index = indexScenarioWithHeuristicAgents;

    fillWithSimpleHeuristicAgents(episode, {
        move: randomRangeFloat(0.4, 0.8),
        aim: {
            aimError: randomRangeFloat(0.5, 2),
            shootChance: randomRangeFloat(0.2, 0.4),
        },
    });

    return episode;
}

