import { createScenarioStatic } from './createScenarioStatic.ts';
import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { SimpleHeuristicAgent } from './Agents/SimpleHeuristicAgent.ts';
import { random, randomRangeFloat } from '../../../../../../lib/random.ts';
import { createScenarioStaticWithCoop } from './createScenarioStaticWithCoop.ts';
import { getScenarioIndex } from './utils.ts';

export const indexScenarioWithMovingAgents = getScenarioIndex();

export async function createScenarioWithMovingAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await (random() > 0.5 ? createScenarioStatic : createScenarioStaticWithCoop)(options);
    episode.index = indexScenarioWithMovingAgents;

    const freeTanks = episode.getFreeTankEids();

    for (const tankEid of freeTanks) {
        const agent = new SimpleHeuristicAgent(tankEid, {
            move: randomRangeFloat(0.2, 1),
        });
        episode.addAgent(tankEid, agent);
    }

    return episode;
}