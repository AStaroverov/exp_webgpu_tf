import { CurrentActorAgent } from '../../../../../tanks/src/Pilots/Agents/CurrentActorAgent.ts';
import { Scenario } from '../types.ts';

export function fillWithCurrentAgents(episode: Scenario) {
    const freeTanks = episode.getFreeTankEids();

    for (const tankEid of freeTanks) {
        const agent = new CurrentActorAgent(tankEid, true);
        episode.setPilot(tankEid, agent);
    }
}

