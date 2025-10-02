import { createBattlefield } from './createBattlefield.ts';
import { createScenarioBase } from './createScenarioBase.ts';
import { Scenario } from './types.ts';
import { fillWithRandomHistoricalAgents } from './Utils/fillWithRandomHistoricalAgents.ts';

export const indexScenarioWithHistoricalAgents = 4;

export async function createScenarioWithHistoricalAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const scenario = await createScenarioBase(options);
    scenario.index = indexScenarioWithHistoricalAgents;
    fillWithRandomHistoricalAgents(scenario);
    return scenario;
}

