import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { randomRangeFloat } from '../../../../../../lib/random.ts';
import { createScenarioWithAlliesActive } from './createScenarioWithAlliesActive.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';

export const indexScenarioWithShootingAgents = 4;

export async function createScenarioWithShootingAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioWithAlliesActive(options);
    episode.index = indexScenarioWithShootingAgents;

    fillWithSimpleHeuristicAgents(episode, {
        aim: {
            aimError: randomRangeFloat(0.05, 0.1),
            shootChance: randomRangeFloat(0.2, 0.4),
        },
    });

    return episode;
}

