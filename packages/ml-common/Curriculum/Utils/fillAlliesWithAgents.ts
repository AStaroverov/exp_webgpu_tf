import { getTankTeamId } from '../../../tanks/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { CurrentActorAgent } from '../../../tanks/src/Pilots/Agents/CurrentActorAgent.ts';
import { Scenario } from '../types.ts';

export function fillAlliesWithAgents(scenario: Scenario, train: boolean) {
    const vehicleEids = scenario.getFreeVehicleEids();
    const firstAgent = scenario.getAlivePilots();
    const activeTeam = getTankTeamId(firstAgent[0].tankEid);

    for (let i = 0; i < vehicleEids.length; i++) {
        const vehicleEid = vehicleEids[i];

        if (getTankTeamId(vehicleEid) !== activeTeam) continue;

        const agent = new CurrentActorAgent(vehicleEid, train);
        scenario.setPilot(vehicleEid, agent);
    }
}

