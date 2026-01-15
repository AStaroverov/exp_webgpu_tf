import { ScenarioCoreOptions } from './createScenarioCore.ts';
import { Scenario } from './types.ts';
import { createRandomNvsMScenario } from './Utils/createRandomNvsMScenario.ts';

/**
 * Simplest scenario: 1 agent vs 1 simple bot at random positions.
 * No fauna, no obstacles - pure 1v1 combat training.
 */
export function createScenario1v1Random(options: ScenarioCoreOptions): Scenario {
    return createRandomNvsMScenario(options, 1, 1);
}
