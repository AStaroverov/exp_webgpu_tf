import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { createScenarioWithAlliesActive } from './createScenarioWithAlliesActive.ts';
import { fillWithBots } from './Utils/fillWithBots.ts';
import { PilotType } from '../../../Pilots/Components/Pilot.ts';

export const indexScenarioWithMovingAgents = 3;

export async function createScenarioWithMovingAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioWithAlliesActive(options);
    episode.index = indexScenarioWithMovingAgents;

    fillWithBots(episode, PilotType.BotOnlyMoving);

    return episode;
}

