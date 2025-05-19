import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { randomRangeFloat } from '../../../../../../lib/random.ts';
import { createScenarioStaticWithCoop } from './createScenarioStaticWithCoop.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';

export const indexScenarioWithShootingAgents = 3;

export async function createScenarioWithShootingAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioStaticWithCoop(options);
    episode.index = indexScenarioWithShootingAgents;

    fillWithSimpleHeuristicAgents(episode, {
        aim: {
            aimError: randomRangeFloat(0.5, 2),
            shootChance: randomRangeFloat(0.2, 0.4),
        },
    });

    return episode;
}

