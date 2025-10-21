import { createScenarioBase } from './createScenarioBase.ts';
import { Scenario } from './types.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';

export const indexStaticScenarioWithBots = 0;

export function createStaticScenarioWithBots(options: Parameters<typeof createScenarioBase>[0]): Scenario {
    const scenario = createScenarioBase({
        ...options,
        alliesCount: 3,
        enemiesCount: 3,
    });

    fillWithSimpleHeuristicAgents(scenario, {
        move: 0.1,
        aim: {
            aimError: 0.3,
            shootChance: 0.02,
        },
    });

    scenario.index = indexStaticScenarioWithBots;
    return scenario;
}