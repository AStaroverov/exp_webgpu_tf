import { ActorAgent } from './Agents/ActorAgent.ts';
import { getTankTeamId } from '../../../ECS/Entities/Tank/TankUtils.ts';
import { random } from '../../../../../../lib/random.ts';
import { createScenarioStatic } from './createScenarioStatic.ts';
import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { getScenarioIndex } from './utils.ts';

export const indexScenarioStaticWithCoop = getScenarioIndex();

export async function createScenarioStaticWithCoop(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const episode = await createScenarioStatic(options);
    episode.index = indexScenarioStaticWithCoop;

    const tankEids = episode.getTankEids();
    const activeTeam = getTankTeamId(episode.getAgents()[0].tankEid);

    const newAgents = [];

    for (let i = 1; i < tankEids.length; i++) {
        const tankEid = tankEids[i];
        if (getTankTeamId(tankEid) !== activeTeam) continue;

        const agent = new ActorAgent(tankEid);
        episode.addAgent(tankEid, agent);
        newAgents.push(agent);

        // with 50% chance, we add one more ally
        if (random() > 0.5) {
            break;
        }
    }

    await Promise.all(newAgents.map(agent => agent.sync?.()));

    return episode;
}