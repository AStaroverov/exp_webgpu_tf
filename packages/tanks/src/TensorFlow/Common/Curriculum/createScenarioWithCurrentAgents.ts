import { createBattlefield } from './createBattlefield.ts';
import { createScenarioBase } from './createScenarioBase.ts';
import { Scenario } from './types.ts';
import { fillWithCurrentAgents } from './Utils/fillWithCurrentAgents.ts';

export const indexScenarioWithCurrentAgents = 5;

export async function createScenarioWithCurrentAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const scenario = await createScenarioBase(options);
    scenario.index = indexScenarioWithCurrentAgents;
    fillWithCurrentAgents(scenario);
    return scenario;
}

