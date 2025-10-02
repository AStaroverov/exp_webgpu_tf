import { SimpleBot, SimpleBotFeatures } from '../../../tanks/src/Pilots/Agents/SimpleBot.ts';
import { Scenario } from '../types.ts';

export function fillWithSimpleHeuristicAgents(episode: Scenario, features: SimpleBotFeatures) {
    const freeTanks = episode.getFreeTankEids();

    for (const tankEid of freeTanks) {
        const agent = new SimpleBot(tankEid, features);
        episode.setPilot(tankEid, agent);
    }
}
