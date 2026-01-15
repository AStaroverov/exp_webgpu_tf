import { createScenarioGridBase } from './createScenarioGridBase.ts';
import { Scenario } from './types.ts';
import { fillWithCurrentAgents } from './Utils/fillWithCurrentAgents.ts';

export function createScenarioWithCurrentAgents(options: Parameters<typeof createScenarioGridBase>[0]): Scenario {
    // Self-play scenario is always train
    options.train = true;
    const scenario = createScenarioGridBase(options);
    fillWithCurrentAgents(scenario);
    return scenario;
}

