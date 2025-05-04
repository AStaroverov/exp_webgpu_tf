import { Scenario } from '../types.ts';
import { RandomHistoricalAgent } from '../Agents/RandomHistoricalAgent.ts';

export async function fillWithRandomHistoricalAgents(episode: Scenario) {
    const freeTanks = episode.getFreeTankEids();
    const newAgents = [];

    for (const tankEid of freeTanks) {
        const agent = new RandomHistoricalAgent(tankEid);
        newAgents.push(agent);
        episode.addAgent(tankEid, agent);
    }

    await Promise.all(newAgents.map(agent => agent.sync?.()));
}

