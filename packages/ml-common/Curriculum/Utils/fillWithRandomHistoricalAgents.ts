import { RandomHistoricalAgent } from '../../../tanks/src/Pilots/Agents/RandomHistoricalAgent.ts';
import { Scenario } from '../types.ts';

export function fillWithRandomHistoricalAgents(episode: Scenario) {
    const freeVehicles = episode.getFreeVehicleEids();

    for (const vehicleEid of freeVehicles) {
        const agent = new RandomHistoricalAgent(vehicleEid);
        episode.setPilot(vehicleEid, agent);
    }
}

