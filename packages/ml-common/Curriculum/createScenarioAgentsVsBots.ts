import { randomRangeFloat } from '../../../lib/random.ts';
import { createScenarioBase } from './createScenarioBase.ts';
import { Scenario } from './types.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';

export function createScenarioAgentsVsBots(level: 0 | 1 | 2, options: Parameters<typeof createScenarioBase>[0]): Scenario {
    const scenario = createScenarioBase(options);
    fillWithSimpleHeuristicAgents(scenario, createBotFeatures(level));
    return scenario;
}

function createBotFeatures(level: 0 | 1 | 2) {
    switch (level) {
        case 0:
            return {
                move: randomRangeFloat(0.05, 0.3),
                aim: {
                    aimError: randomRangeFloat(0.5, 0.8),
                    shootChance: randomRangeFloat(0.01, 0.3),
                },
            };
        case 1:
            return {
                move: randomRangeFloat(0.1, 0.7),
                aim: {
                    aimError: randomRangeFloat(0.2, 0.6),
                    shootChance: randomRangeFloat(0.2, 0.6),
                },
            };
        case 2:
            return {
                move: randomRangeFloat(0.5, 1),
                aim: {
                    aimError: randomRangeFloat(0.05, 0.3),
                    shootChance: randomRangeFloat(0.5, 1),
                },
            };
        default:
            throw new Error('Not implemented');
    }
}