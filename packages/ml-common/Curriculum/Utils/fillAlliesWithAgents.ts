import { getTankTeamId } from '../../../tanks/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { CurrentActorAgent } from '../../Agents/CurrentActorAgent.ts';
import { Scenario } from '../types.ts';

export function fillAlliesWithAgents(scenario: Scenario) {
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

