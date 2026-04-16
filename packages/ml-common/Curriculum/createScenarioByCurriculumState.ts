import { createScenario1v1Random } from './createScenario1v1Random.ts';
import { createScenarioDiagonalWall } from './createScenarioDiagonalWall.ts';
import { createScenarioAgentsVsBots1 } from './createScenarioAgentsVsBots1.ts';
import { createScenarioGridBase } from './createScenarioGridBase.ts';
import { createScenarioWithHistoricalAgents as createScenarioFrozenSelfPlay } from './createScenarioWithHistoricalAgents.ts';
import { createStaticScenarioAgentsVsBots0 } from './createStaticScenarioAgentsVsBots0.ts';
import { CurriculumState, Scenario } from './types.ts';
import { createScenarioWithCurrentAgents as createScenarioSelfPlay } from './createScenarioWithCurrentAgents.ts';
import { createScenario3v3Random } from './createScenario3v3Random.ts';

type ScenarioOptions = Parameters<typeof createScenarioGridBase>[0];

const scenarios = [
    createScenario1v1Random,                    // 0: 1v1 random positions, simplest bot
    createScenario3v3Random.bind(null, 0),      // 1
    createScenarioDiagonalWall,                 // 2: 1v1 diagonal with 3-building wall
    createStaticScenarioAgentsVsBots0,          // 3
    createScenario3v3Random.bind(null, 1),      // 4
    createScenarioAgentsVsBots1,                // 5
    createScenarioFrozenSelfPlay,               // 6
    createScenarioSelfPlay,                     // 7
] as const;

export const scenariosCount = scenarios.length;

const passThreshold = 0.7;

export async function createScenarioByCurriculumState(curriculumState: CurriculumState, options: Omit<ScenarioOptions, 'index'>): Promise<Scenario> {
    const constructorOptions = options as ScenarioOptions;

    // Find the first scenario that hasn't reached the pass threshold.
    // Previous scenarios are skipped — only the current one is played.
    let currentIndex = 0;
    for (let i = 0; i < scenarios.length - 1; i++) {
        const successRatio = curriculumState.mapScenarioIndexToSuccessRatio[i] ?? 0;
        if (successRatio < passThreshold) {
            break;
        }
        currentIndex = i + 1;
    }

    constructorOptions.index = currentIndex;
    const constructor = scenarios[currentIndex];

    return constructor(constructorOptions);
}
