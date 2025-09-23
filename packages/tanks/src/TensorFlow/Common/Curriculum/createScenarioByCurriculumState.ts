import { max, min } from '../../../../../../lib/math.ts';
import { random } from '../../../../../../lib/random.ts';
import { CurriculumState } from '../../PPO/channels.ts';
import { createBattlefield } from './createBattlefield.ts';
import { createScenarioAgentsVsBots, indexScenarioAgentsVsBots } from './createScenarioAgentsVsBots.ts';
import { createScenarioAgentsVsBots1, indexScenarioAgentsVsBots1 } from './createScenarioAgentsVsBots1.ts';
import { createScenarioAgentsVsBots2, indexScenarioAgentsVsBots2 } from './createScenarioAgentsVsBots2.ts';
import {
    createScenarioWithHistoricalAgents,
    indexScenarioWithHistoricalAgents,
} from './createScenarioWithHistoricalAgents.ts';
import { createStaticScenarioWithBots, indexStaticScenarioWithBots } from './createStaticScenarioWithBots.ts';
import { Scenario } from './types.ts';

type ScenarioOptions = Parameters<typeof createBattlefield>[0];

const mapEntries = [
    [indexStaticScenarioWithBots, createStaticScenarioWithBots],
    [indexScenarioAgentsVsBots, createScenarioAgentsVsBots.bind(null, 0)],
    [indexScenarioAgentsVsBots1, createScenarioAgentsVsBots1],
    [indexScenarioAgentsVsBots2, createScenarioAgentsVsBots2],
    [indexScenarioWithHistoricalAgents, createScenarioWithHistoricalAgents],
] as const;
const mapIndexToConstructor = new Map<number, (options: ScenarioOptions) => Promise<Scenario>>(mapEntries);

if (mapIndexToConstructor.size !== mapEntries.length) {
    throw new Error('Scenario index is not unique');
}

export const scenariosCount = mapIndexToConstructor.size;

export async function createScenarioByCurriculumState(curriculumState: CurriculumState, options: ScenarioOptions): Promise<Scenario> {
    let constructor = createStaticScenarioWithBots;

    let weights = [];
    let totalWeight = 0;
    for (let i = 0, minSuccessRatio = 1; i < mapIndexToConstructor.size; i++) {
        let successRatio: number | undefined = curriculumState.mapScenarioIndexToSuccessRatio[i];

        // Unlock next scenarios when all previous reach at least 0.5 avg success
        if (successRatio === undefined && minSuccessRatio < 0.5) {
            break;
        }

        successRatio ??= 0;

        const weight = max(0.01, 0.95 - successRatio);

        weights.push(weight);
        totalWeight += weight;
        minSuccessRatio = min(minSuccessRatio, successRatio);
    }

    for (let i = 0, r = random() * totalWeight; i < weights.length; i++) {
        const weight = weights[i];
        if (r < weight) {
            constructor = mapIndexToConstructor.get(i) ?? (() => {
                console.warn(`Scenario ${i} not found, using default scenario static`);
                return createStaticScenarioWithBots;
            })();
            break;
        }
        r -= weight;
    }

    return constructor(options);
}
