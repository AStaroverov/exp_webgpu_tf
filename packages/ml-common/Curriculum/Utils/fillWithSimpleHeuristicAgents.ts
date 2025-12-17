import { SimpleBot, SimpleBotFeatures } from '../../../tanks/src/Pilots/Agents/SimpleBot.ts';
import { Scenario } from '../types.ts';

export function fillWithSimpleHeuristicAgents(episode: Scenario, features: SimpleBotFeatures) {
    const freeVehicles = episode.getFreeVehicleEids();

    for (const vehicleEid of freeVehicles) {
        const agent = new SimpleBot(vehicleEid, features);
        episode.setPilot(vehicleEid, agent);
    }
}
