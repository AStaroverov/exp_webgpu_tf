import { Scenario } from './types.ts';
import { SimpleHeuristicAgent, SimpleHeuristicAgentFeatures } from './Agents/SimpleHeuristicAgent.ts';

export function addSimpleHeuristicAgents(episode: Scenario, features: SimpleHeuristicAgentFeatures) {
    const freeTanks = episode.getFreeTankEids();

    for (const tankEid of freeTanks) {
        const agent = new SimpleHeuristicAgent(tankEid, features);
        episode.addAgent(tankEid, agent);
    }
}

