import { clamp } from 'lodash';
import { min } from '../../../lib/math.ts';
import { random } from '../../../lib/random.ts';
import { createScenarioAgentsVsBots1 } from './createScenarioAgentsVsBots1.ts';
import { createScenarioBase } from './createScenarioBase.ts';
import { createScenarioWithHistoricalAgents as createScenarioFrozenSelfPlay } from './createScenarioWithHistoricalAgents.ts';
import { createStaticScenarioAgentsVsBots0 } from './createStaticScenarioAgentsVsBots0.ts';
import { CurriculumState, Scenario } from './types.ts';
import { createScenarioWithCurrentAgents as createScenarioSelfPlay } from './createScenarioWithCurrentAgents.ts';

type ScenarioOptions = Parameters<typeof createScenarioBase>[0];

const mapEntries = [
    [0, createStaticScenarioAgentsVsBots0],
    [1, createScenarioAgentsVsBots1],
    [2, createScenarioFrozenSelfPlay],
    [3, createScenarioSelfPlay],
] as const;
const mapIndexToConstructor = new Map<number, (options: ScenarioOptions) => Scenario>(mapEntries);

if (mapIndexToConstructor.size !== mapEntries.length) {
    throw new Error('Scenario index is not unique');
}

export const scenariosCount = mapIndexToConstructor.size;

const edge = 0.3; // success ratio to unlock next scenario

export async function createScenarioByCurriculumState(curriculumState: CurriculumState, options: Omit<ScenarioOptions, 'index'>): Promise<Scenario> {
    const constructorOptions = options as ScenarioOptions;

    if (random() < 0.5) {
        constructorOptions.index = 3
        return createScenarioSelfPlay(constructorOptions);
    }

    let constructor = createStaticScenarioAgentsVsBots0;

    let weights = [];
    let totalWeight = 0;
    for (let i = 0, minSuccessRatio = 1; i < mapIndexToConstructor.size; i++) {
        let successRatio: number | undefined = curriculumState.mapScenarioIndexToSuccessRatio[i];

        // Unlock next scenarios when all previous reach at least 0.3 avg success
        if (successRatio === undefined && minSuccessRatio < edge) {
            break;
        }

        successRatio ??= 0;

        const weight = clamp(1 - successRatio, 0.2, 1);

        weights.push(weight);
        totalWeight += weight;
        minSuccessRatio = min(minSuccessRatio, successRatio);
    }

    for (let i = 0, r = random() * totalWeight; i < weights.length; i++) {
        const weight = weights[i];
        if (r < weight) {
            constructorOptions.index = i;
            constructor = mapIndexToConstructor.get(i) ?? (() => {
                console.error(`Scenario ${i} not found, using default scenario static`);
                constructorOptions.index = 0;
                return createStaticScenarioAgentsVsBots0;
            })();
            break;
        }
        r -= weight;
    }

    return constructor(constructorOptions);
}
