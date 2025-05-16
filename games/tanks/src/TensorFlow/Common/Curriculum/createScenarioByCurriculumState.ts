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

    for (const [index, createScenario] of mapIndexToConstructor.entries()) {
        const successRatio = curriculumState.mapScenarioIndexToSuccessRatio[Number(index)] ?? 0;
        const weight = 1 - successRatio;

        weights.push({ createScenario, weight });
        totalWeight += weight;

        if (successRatio < 0.75) break;
    }

    let r = random() * totalWeight;

    for (const { createScenario, weight } of weights) {
        if (r < weight) {
            constructor = createScenario;
            break;
        }
        r -= weight;
    }

    return constructor(options);
}
