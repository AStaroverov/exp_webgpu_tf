import { Scenario } from '../types.ts';
import { RandomHistoricalAgent } from '../../../../Pilots/Agents/RandomHistoricalAgent.ts';

export function fillWithRandomHistoricalAgents(episode: Scenario) {
    const freeTanks = episode.getFreeTankEids();

    for (const tankEid of freeTanks) {
        const agent = new RandomHistoricalAgent(tankEid);
        episode.setPilot(tankEid, agent);
    }
}

