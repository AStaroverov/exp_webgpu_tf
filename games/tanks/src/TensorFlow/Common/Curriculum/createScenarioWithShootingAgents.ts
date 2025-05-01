import { createScenarioStatic } from './createScenarioStatic.ts';
import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { random, randomRangeFloat } from '../../../../../../lib/random.ts';
import { createScenarioStaticWithCoop } from './createScenarioStaticWithCoop.ts';
import { getScenarioIndex } from './utils.ts';
import { addSimpleHeuristicAgents } from './addSimpleHeuristicAgents.ts';

export const indexScenarioWithShootingAgents = getScenarioIndex();

export async function createScenarioWithShootingAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await (random() > 0.5 ? createScenarioStatic : createScenarioStaticWithCoop)(options);
    episode.index = indexScenarioWithShootingAgents;

    addSimpleHeuristicAgents(episode, {
        aim: {
            aimError: randomRangeFloat(1, 3),
            shootChance: randomRangeFloat(0.2, 0.4),
        },
    });

    return episode;
}

