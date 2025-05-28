import { Scenario } from '../types.ts';
import { SimpleBot, SimpleBotFeatures } from '../../../../Pilots/Agents/SimpleBot.ts';

export function fillWithSimpleHeuristicAgents(episode: Scenario, features: SimpleBotFeatures) {
    const freeTanks = episode.getFreeTankEids();

    for (const tankEid of freeTanks) {
        const agent = new SimpleBot(tankEid, features);
        episode.setPilot(tankEid, agent);
    }
}

