import { RandomHistoricalAgent } from '../../../tanks/src/Pilots/Agents/RandomHistoricalAgent.ts';
import { getFreeVehicaleEids, Pilot } from '../../../tanks/src/Pilots/Components/Pilot.ts';
import { Scenario } from '../types.ts';

export function fillWithRandomHistoricalAgents(scenario: Scenario) {
    const freeVehicles = getFreeVehicaleEids();

    for (const vehicleEid of freeVehicles) {
        const agent = new RandomHistoricalAgent(vehicleEid);
        Pilot.addComponent(scenario.world, vehicleEid, agent);
    }
}

