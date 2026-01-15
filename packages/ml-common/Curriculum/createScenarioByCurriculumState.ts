import { clamp } from 'lodash';
import { min } from '../../../lib/math.ts';
import { random } from '../../../lib/random.ts';
import { createScenario1v1Random } from './createScenario1v1Random.ts';
import { createScenario3v3Random } from './createScenario3v3Random.ts';
import { createScenarioDiagonal } from './createScenarioDiagonal.ts';
import { createScenarioDiagonalWall } from './createScenarioDiagonalWall.ts';
import { createScenarioAgentsVsBots1 } from './createScenarioAgentsVsBots1.ts';
import { createScenarioGridBase } from './createScenarioGridBase.ts';
import { createScenarioWithHistoricalAgents as createScenarioFrozenSelfPlay } from './createScenarioWithHistoricalAgents.ts';
import { createStaticScenarioAgentsVsBots0 } from './createStaticScenarioAgentsVsBots0.ts';
import { CurriculumState, Scenario } from './types.ts';
import { createScenarioWithCurrentAgents as createScenarioSelfPlay } from './createScenarioWithCurrentAgents.ts';

type ScenarioOptions = Parameters<typeof createScenarioGridBase>[0];

const mapEntries = [
    createScenario1v1Random,       // 1v1 random positions, simplest bot
    createScenarioDiagonal,        // 1v1 diagonal with center obstacle
    createScenario3v3Random.bind(null, 0),
    createScenarioDiagonalWall,    // 1v1 diagonal with 3-building wall
    createStaticScenarioAgentsVsBots0,
    createScenario3v3Random.bind(null, 1),
    createScenarioAgentsVsBots1,
    createScenarioFrozenSelfPlay,
    createScenarioSelfPlay,
].map((constructor, index) => ([index, constructor] as const));
const mapIndexToConstructor = new Map<number, (options: ScenarioOptions) => Scenario>(mapEntries);

if (mapIndexToConstructor.size !== mapEntries.length) {
    throw new Error('Scenario index is not unique');
}

export const scenariosCount = mapIndexToConstructor.size;

const edge = 0.5; // success ratio to unlock next scenario

export async function createScenarioByCurriculumState(curriculumState: CurriculumState, options: Omit<ScenarioOptions, 'index'>): Promise<Scenario> {
    const constructorOptions = options as ScenarioOptions;

    let constructor = createScenario1v1Random;

    let weights = [];
    let totalWeight = 0;
    for (let i = 0, minSuccessRatio = 1; i < mapIndexToConstructor.size; i++) {
        let successRatio: number | undefined = curriculumState.mapScenarioIndexToSuccessRatio[i];

        // Unlock next scenarios when all previous reach at least 0.3 avg success
        if (successRatio === undefined && minSuccessRatio < edge) {
            break;
        }

        successRatio ??= 0;

        const weight = clamp(0.9 - successRatio, 0.2, 1);

        weights.push(weight);
        totalWeight += weight;
        minSuccessRatio = min(minSuccessRatio, successRatio);
    }

    for (let i = 0, r = random() * totalWeight; i < weights.length; i++) {
        const weight = weights[i];
        if (r < weight) {
            constructorOptions.index = i;
            constructor = mapIndexToConstructor.get(i) ?? (() => {
                console.error(`Scenario ${i} not found, using default scenario 1v1`);
                constructorOptions.index = 0;
                return createScenario1v1Random;
            })();
            break;
        }
        r -= weight;
    }

    return constructor(constructorOptions);
}
