import { createScenarioGridBase } from './createScenarioGridBase.ts';
import { Scenario } from './types.ts';
import { fillWithRandomHistoricalAgents } from './Utils/fillWithRandomHistoricalAgents.ts';

export function createScenarioWithHistoricalAgents(options: Parameters<typeof createScenarioGridBase>[0]): Scenario {
    const scenario = createScenarioGridBase(options);
    fillWithRandomHistoricalAgents(scenario);
    return scenario;
}

