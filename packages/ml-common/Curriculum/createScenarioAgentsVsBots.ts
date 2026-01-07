import { randomRangeFloat } from '../../../lib/random.ts';
import { createScenarioGridBase } from './createScenarioGridBase.ts';
import { Scenario } from './types.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';

export function createScenarioAgentsVsBots(level: 0 | 1 | 2, options: Parameters<typeof createScenarioGridBase>[0]): Scenario {
    const scenario = createScenarioGridBase(options);
    fillWithSimpleHeuristicAgents(scenario, createBotFeatures(level));
    return scenario;
}

function createBotFeatures(level: 0 | 1 | 2) {
    switch (level) {
        case 0:
            return {
                move: randomRangeFloat(0.1, 0.5),
                aim: {
                    aimError: randomRangeFloat(0.5, 0.8),
                    shootChance: randomRangeFloat(0.01, 0.3),
                },
            };
        case 1:
            return {
                move: randomRangeFloat(0.5, 1),
                aim: {
                    aimError: randomRangeFloat(0.3, 0.5),
                    shootChance: randomRangeFloat(0.3, 0.8),
                },
            };
        default:
            throw new Error('Not implemented');
    }
}