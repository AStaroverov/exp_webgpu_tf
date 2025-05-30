import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { createScenarioWithAlliesActive } from './createScenarioWithAlliesActive.ts';
import { fillWithBots } from './Utils/fillWithBots.ts';
import { PilotType } from '../../../Pilots/Components/Pilot.ts';

export const indexScenarioWithShootingAgents = 4;

export async function createScenarioWithShootingAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioWithAlliesActive(options);
    episode.index = indexScenarioWithShootingAgents;

    fillWithBots(episode, PilotType.BotOnlyShooting);

    return episode;
}

