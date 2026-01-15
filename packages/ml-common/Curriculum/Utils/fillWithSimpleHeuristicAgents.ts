import { SimpleBot, SimpleBotFeatures } from '../../../tanks/src/Plugins/Pilots/Agents/SimpleBot.ts';
import { getFreeVehicaleEids, Pilot } from '../../../tanks/src/Plugins/Pilots/Components/Pilot.ts';
import { Scenario } from '../types.ts';

export function fillWithSimpleHeuristicAgents(scenario: Scenario, features: SimpleBotFeatures) {
    const freeVehicles = getFreeVehicaleEids();

    for (const vehicleEid of freeVehicles) {
        const agent = new SimpleBot(vehicleEid, features);
        Pilot.addComponent(scenario.world, vehicleEid, agent);
    }
}
