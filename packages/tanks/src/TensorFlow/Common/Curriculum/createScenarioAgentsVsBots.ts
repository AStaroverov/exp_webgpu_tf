import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { fillAlliesWithAgents } from './Utils/fillAlliesWithAgents.ts';
import { createScenarioBase } from './createScenarioBase.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';
import { randomRangeFloat } from '../../../../../../lib/random.ts';

export const indexScenarioAgentsVsBots = 1;

export async function createScenarioAgentsVsBots(level: 0 | 1 | 2, options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const scenario = await createScenarioBase(options);
    scenario.index = indexScenarioAgentsVsBots;

    fillAlliesWithAgents(scenario);
    fillWithSimpleHeuristicAgents(scenario, createBotFeatures(level));

    return scenario;
}

function createBotFeatures(level: 0 | 1 | 2) {
    switch (level) {
        case 0:
            return {
                move: randomRangeFloat(0.1, 0.3),
                aim: {
                    aimError: randomRangeFloat(0.3, 0.5),
                    shootChance: randomRangeFloat(0.05, 0.2),
                },
            };
        case 1:
            return {
                move: randomRangeFloat(0.3, 0.8),
                aim: {
                    aimError: randomRangeFloat(0.1, 0.3),
                    shootChance: randomRangeFloat(0.2, 0.6),
                },
            };
        case 2:
            return {
                move: randomRangeFloat(0.8, 1),
                aim: {
                    aimError: randomRangeFloat(0, 0.1),
                    shootChance: randomRangeFloat(0.6, 1),
                },
            };
        default:
            throw new Error('Not implemented');
    }
}