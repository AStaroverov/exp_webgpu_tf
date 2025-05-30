import { Scenario } from '../types.ts';
import { PilotType } from '../../../../Pilots/Components/Pilot.ts';

export function fillWithRandomHistoricalAgents(episode: Scenario) {
    const freeTanks = episode.getFreeTankEids();

    for (const tankEid of freeTanks) {
        episode.setPilot(tankEid, PilotType.AgentRandom);
    }
}

