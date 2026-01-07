import { getTankTeamId } from '../../../tanks/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { CurrentActorAgent } from '../../../tanks/src/Pilots/Agents/CurrentActorAgent.ts';
import { getRegistratedAgents, getFreeVehicaleEids, Pilot } from '../../../tanks/src/Pilots/Components/Pilot.ts';
import { Scenario } from '../types.ts';

export function fillAlliesWithAgents(scenario: Scenario) {
    const vehicleEids = getFreeVehicaleEids();
    const firstAgent = getRegistratedAgents();
    const activeTeam = getTankTeamId(firstAgent[0].tankEid);

    for (let i = 0; i < vehicleEids.length; i++) {
        const vehicleEid = vehicleEids[i];

        if (getTankTeamId(vehicleEid) !== activeTeam) continue;

        const agent = new CurrentActorAgent(vehicleEid, scenario.train);
        Pilot.addComponent(scenario.world, vehicleEid, agent);
    }
}

