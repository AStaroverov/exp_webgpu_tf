import { TankAgent } from '../../PPO/Actor/ActorAgent.ts';
import { getTankTeamId } from '../../../ECS/Entities/Tank/TankUtils.ts';
import { random } from '../../../../../../lib/random.ts';
import { EntityId } from 'bitecs';
import { createScenario0 } from './createScenario0.ts';
import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';

export async function createScenario1(
    createAgent: (tankEid: EntityId) => TankAgent,
    options: Parameters<typeof createBattlefield>[0],
): Promise<Scenario> {
    const episode = await createScenario0(createAgent, options);
    const tankEids = episode.getTankEids();
    const activeTeam = getTankTeamId(episode.getAgents()[0].tankEid);

    const newAgents = [];

    for (let i = 1; i < tankEids.length; i++) {
        const tankEid = tankEids[i];
        if (getTankTeamId(tankEid) !== activeTeam) continue;

        const agent = createAgent(tankEid);
        episode.addAgent(tankEid, agent);
        newAgents.push(agent);

        // with 50% chance, we add one more ally
        if (random() > 0.5) {
            break;
        }
    }

    await Promise.all(newAgents.map(agent => agent.sync()));

    episode.index = 1;

    return episode;
}