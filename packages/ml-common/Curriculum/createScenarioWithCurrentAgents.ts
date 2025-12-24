import { createScenarioBase } from './createScenarioBase.ts';
import { Scenario } from './types.ts';
import { fillWithCurrentAgents } from './Utils/fillWithCurrentAgents.ts';

export function createScenarioWithCurrentAgents(options: Parameters<typeof createScenarioBase>[0]): Scenario {
    // Self-play scenario is always train
    options.train = true;
    const scenario = createScenarioBase(options);
    fillWithCurrentAgents(scenario);
    return scenario;
}

