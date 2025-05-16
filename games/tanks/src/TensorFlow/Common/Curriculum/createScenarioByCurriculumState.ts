import { Scenario } from './types.ts';
import { CurriculumState } from '../../PPO/channels.ts';
import { createScenarioStatic, indexScenarioStatic } from './createScenarioStatic.ts';
import { random } from '../../../../../../lib/random.ts';
import { createScenarioStaticWithCoop, indexScenarioStaticWithCoop } from './createScenarioStaticWithCoop.ts';
import { createBattlefield } from './createBattlefield.ts';
import { createScenarioWithMovingAgents, indexScenarioWithMovingAgents } from './createScenarioWithMovingAgents.ts';
import {
    createScenarioWithShootingAgents,
    indexScenarioWithShootingAgents,
} from './createScenarioWithShootingAgents.ts';
import {
    createScenarioWithHeuristicAgents,
    indexScenarioWithHeuristicAgents,
} from './createScenarioWithHeuristicAgents.ts';
import {
    createScenarioWithHistoricalAgents,
    indexScenarioWithHistoricalAgents,
} from './createScenarioWithHistoricalAgents.ts';
import { min } from '../../../../../../lib/math.ts';

type ScenarioOptions = Parameters<typeof createBattlefield>[0];

const mapEntries = [
    [indexScenarioStatic, createScenarioStatic],
    [indexScenarioStaticWithCoop, createScenarioStaticWithCoop],
    [indexScenarioWithMovingAgents, createScenarioWithMovingAgents],
    [indexScenarioWithShootingAgents, createScenarioWithShootingAgents],
    [indexScenarioWithHeuristicAgents, createScenarioWithHeuristicAgents],
    [indexScenarioWithHistoricalAgents, createScenarioWithHistoricalAgents],
] as const;
const mapIndexToConstructor = new Map(mapEntries);

if (mapIndexToConstructor.size !== mapEntries.length) {
    throw new Error('Scenario index is not unique');
}

export const scenariosCount = mapIndexToConstructor.size;

export async function createScenarioByCurriculumState(curriculumState: CurriculumState, options: ScenarioOptions): Promise<Scenario> {
    let constructor = createScenarioStatic;

    let weights = [];
    let totalWeight = 0;
    for (let i = 0, minSuccessRatio = 1; i < mapIndexToConstructor.size; i++) {
        const successRatio = curriculumState.mapScenarioIndexToSuccessRatio[i] ?? 0;
        if (successRatio === 0 && minSuccessRatio < 0.75) {
            break;
        }

        const weight = 1 - successRatio;

        weights.push(weight);
        totalWeight += weight;
        minSuccessRatio = min(minSuccessRatio, successRatio);
    }

    for (let i = 0, r = random() * totalWeight; i < weights.length; i++) {
        const weight = weights[i];
        if (r < weight) {
            constructor = mapIndexToConstructor.get(i) ?? (() => {
                console.warn(`Scenario ${ i } not found, using default scenario static`);
                return createScenarioStatic;
            })();
            break;
        }
        r -= weight;
    }

    return constructor(options);
}
