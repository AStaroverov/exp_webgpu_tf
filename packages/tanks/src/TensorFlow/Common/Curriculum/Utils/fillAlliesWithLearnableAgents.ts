import { Scenario } from '../types.ts';
import { getTankTeamId } from '../../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { PilotType } from '../../../../Pilots/Components/Pilot.ts';

export function fillAlliesWithLearnableAgents(scenario: Scenario) {
    const tankEids = scenario.getFreeTankEids();
    const firstAgent = scenario.getAlivePilots();
    const activeTeam = getTankTeamId(firstAgent[0].tankEid);

    for (let i = 0; i < tankEids.length; i++) {
        const tankEid = tankEids[i];

        if (getTankTeamId(tankEid) !== activeTeam) continue;

        scenario.setPilot(tankEid, PilotType.AgentLearnable);
    }
}

