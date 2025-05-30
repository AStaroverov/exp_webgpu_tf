import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { createScenarioWithAlliesActive } from './createScenarioWithAlliesActive.ts';
import { fillWithBots } from './Utils/fillWithBots.ts';
import { PilotType } from '../../../Pilots/Components/Pilot.ts';

export const indexScenarioWithStrongHeuristicAgents = 6;

export async function createScenarioWithStrongHeuristicAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioWithAlliesActive(options);
    episode.index = indexScenarioWithStrongHeuristicAgents;

    fillWithBots(episode, PilotType.BotStrong);

    return episode;
}

