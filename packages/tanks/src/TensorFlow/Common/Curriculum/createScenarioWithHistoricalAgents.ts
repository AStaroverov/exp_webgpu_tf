import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { fillWithRandomHistoricalAgents } from './Utils/fillWithRandomHistoricalAgents.ts';
import { createScenarioBase } from './createScenarioBase.ts';
import { fillAlliesWithAgents } from './Utils/fillAlliesWithAgents.ts';

export const indexScenarioWithHistoricalAgents = 4;

export async function createScenarioWithHistoricalAgents(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const scenario = await createScenarioBase(options);
    scenario.index = indexScenarioWithHistoricalAgents;

    fillAlliesWithAgents(scenario);
    fillWithRandomHistoricalAgents(scenario);

    return scenario;
}

