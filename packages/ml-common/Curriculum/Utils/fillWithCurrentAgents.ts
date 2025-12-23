import { CurrentActorAgent } from '../../../tanks/src/Pilots/Agents/CurrentActorAgent.ts';
import { getFreeVehicaleEids, Pilot } from '../../../tanks/src/Pilots/Components/Pilot.ts';
import { Scenario } from '../types.ts';

export function fillWithCurrentAgents(scenario: Scenario) {
    const freeVehicles = getFreeVehicaleEids();

    for (const vehicleEid of freeVehicles) {
        const agent = new CurrentActorAgent(vehicleEid, false);
        Pilot.addComponent(scenario.world, vehicleEid, agent);
    }
}

