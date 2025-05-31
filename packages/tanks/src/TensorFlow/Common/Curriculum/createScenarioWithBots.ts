import { Scenario } from './types.ts';
import { createScenarioBase } from './createScenarioBase.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';

export const indexScenarioWithBots = 0;

export async function createScenarioWithBots(options: Parameters<typeof createScenarioBase>[0]): Promise<Scenario> {
    const scenario = await createScenarioBase({
        ...options,
    });

    fillWithSimpleHeuristicAgents(scenario, {
        move: 0.1,
        aim: {
            aimError: 0.3,
            shootChance: 0.02,
        },
    });

    scenario.index = indexScenarioWithBots;
    return scenario;
}