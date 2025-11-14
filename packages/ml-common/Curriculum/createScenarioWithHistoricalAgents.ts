import { createBattlefield } from './createBattlefield.ts';
import { createScenarioBase } from './createScenarioBase.ts';
import { Scenario } from './types.ts';
import { fillWithRandomHistoricalAgents } from './Utils/fillWithRandomHistoricalAgents.ts';

export const indexScenarioWithHistoricalAgents = 3;

export function createScenarioWithHistoricalAgents(options: Parameters<typeof createBattlefield>[0]): Scenario {
    const scenario = createScenarioBase(options);
    scenario.index = indexScenarioWithHistoricalAgents;
    fillWithRandomHistoricalAgents(scenario);
    return scenario;
}

