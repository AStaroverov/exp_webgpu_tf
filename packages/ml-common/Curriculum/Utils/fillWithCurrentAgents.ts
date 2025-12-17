import { CurrentActorAgent } from '../../../tanks/src/Pilots/Agents/CurrentActorAgent.ts';
import { Scenario } from '../types.ts';

export function fillWithCurrentAgents(episode: Scenario) {
    const freeVehicles = episode.getFreeVehicleEids();

    for (const vehicleEid of freeVehicles) {
        const agent = new CurrentActorAgent(vehicleEid, false);
        episode.setPilot(vehicleEid, agent);
    }
}

