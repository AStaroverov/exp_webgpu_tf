import { Scenario } from '../types.ts';
import { getTankTeamId } from '../../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { CurrentActorAgent } from '../../../../Pilots/Agents/CurrentActorAgent.ts';

export function fillAlliesWithActorAgents(scenario: Scenario) {
    const tankEids = scenario.getFreeTankEids();
    const firstAgent = scenario.getAlivePilots();
    const activeTeam = getTankTeamId(firstAgent[0].tankEid);

    for (let i = 0; i < tankEids.length; i++) {
        const tankEid = tankEids[i];

        if (getTankTeamId(tankEid) !== activeTeam) continue;

        const agent = new CurrentActorAgent(tankEid, true);
        scenario.setPilot(tankEid, agent);
    }
}

