import { ScenarioCoreOptions } from './createScenarioCore.ts';
import { Scenario } from './types.ts';
import { BotLevel } from './Utils/botFeatures.ts';
import { createRandomNvsMScenario } from './Utils/createRandomNvsMScenario.ts';

/**
 * 3 agents vs 3 simple bots at random positions.
 * Team-based combat training in random environment.
 */
export function createScenario3v3Random(botLevel: BotLevel = 0, options: ScenarioCoreOptions): Scenario {
    return createRandomNvsMScenario(options, 3, 3, botLevel);
}

