import { Scenario } from '../types.ts';
import { getTankTeamId } from '../../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { CurrentActorAgent, TankAgent } from '../Agents/CurrentActorAgent.ts';

export async function fillAlliesWithActorAgents(episode: Scenario) {
    const tankEids = episode.getFreeTankEids();
    const firstAgent = episode.getAgents();
    const activeTeam = getTankTeamId(firstAgent[0].tankEid);

    const newAgents: TankAgent[] = [];

    for (let i = 0; i < tankEids.length; i++) {
        const tankEid = tankEids[i];

        if (getTankTeamId(tankEid) !== activeTeam) continue;

        const agent = new CurrentActorAgent(tankEid, true);
        episode.addAgent(tankEid, agent);
        newAgents.push(agent);
    }

    await Promise.all(newAgents.map(agent => agent.sync?.()));

    return () => {
        newAgents.forEach(agent => agent.dispose?.());
    };
}

