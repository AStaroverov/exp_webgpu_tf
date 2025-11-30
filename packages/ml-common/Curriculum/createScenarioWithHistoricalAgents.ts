import { createScenarioBase } from './createScenarioBase.ts';
import { Scenario } from './types.ts';
import { fillWithRandomHistoricalAgents } from './Utils/fillWithRandomHistoricalAgents.ts';

export function createScenarioWithHistoricalAgents(options: Parameters<typeof createScenarioBase>[0]): Scenario {
    const scenario = createScenarioBase(options);
    fillWithRandomHistoricalAgents(scenario);
    return scenario;
}

