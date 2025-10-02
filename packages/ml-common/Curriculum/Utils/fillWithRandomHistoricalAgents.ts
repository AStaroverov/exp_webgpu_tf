import { RandomHistoricalAgent } from '../../../tanks/src/Pilots/Agents/RandomHistoricalAgent.ts';
import { Scenario } from '../types.ts';

export function fillWithRandomHistoricalAgents(episode: Scenario) {
    const freeTanks = episode.getFreeTankEids();

    for (const tankEid of freeTanks) {
        const agent = new RandomHistoricalAgent(tankEid);
        episode.setPilot(tankEid, agent);
    }
}

