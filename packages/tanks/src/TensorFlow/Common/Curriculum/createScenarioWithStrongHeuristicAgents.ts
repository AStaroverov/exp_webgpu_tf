import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { createScenarioWithAlliesActive } from './createScenarioWithAlliesActive.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';

export const indexScenarioWithStrongHeuristicAgents = 6;

export async function createScenarioWithStrongHeuristicAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioWithAlliesActive(options);
    episode.index = indexScenarioWithStrongHeuristicAgents;

    fillWithSimpleHeuristicAgents(episode, {
        move: 1,
        aim: {
            aimError: 0,
            shootChance: 0.8,
        },
    });

    return episode;
}

