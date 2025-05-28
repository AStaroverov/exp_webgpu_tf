import { Scenario } from './types.ts';
import { CurriculumState } from '../../PPO/channels.ts';
import { createScenarioWithAlliesStatic, indexScenarioWithAlliesStatic } from './createScenarioWithAlliesStatic.ts';
import { random } from '../../../../../../lib/random.ts';
import { createScenarioWithAlliesActive, indexScenarioWithAlliesActive } from './createScenarioWithAlliesActive.ts';
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
    createScenarioWithStrongHeuristicAgents,
    indexScenarioWithStrongHeuristicAgents,
} from './createScenarioWithStrongHeuristicAgents.ts';
import {
    createScenarioWithHistoricalAgents,
    indexScenarioWithHistoricalAgents,
} from './createScenarioWithHistoricalAgents.ts';
import { max, min } from '../../../../../../lib/math.ts';
import { createScenarioSoloStatic, indexScenarioSoloStatic } from './createScenarioSoloStatic.ts';

type ScenarioOptions = Parameters<typeof createBattlefield>[0];

const mapEntries = [
    [indexScenarioSoloStatic, createScenarioSoloStatic],
    [indexScenarioWithAlliesStatic, createScenarioWithAlliesStatic],
    [indexScenarioWithAlliesActive, createScenarioWithAlliesActive],
    [indexScenarioWithMovingAgents, createScenarioWithMovingAgents],
    [indexScenarioWithShootingAgents, createScenarioWithShootingAgents],
    [indexScenarioWithHeuristicAgents, createScenarioWithHeuristicAgents],
    [indexScenarioWithStrongHeuristicAgents, createScenarioWithStrongHeuristicAgents],
    [indexScenarioWithHistoricalAgents, createScenarioWithHistoricalAgents],
] as const;
const mapIndexToConstructor = new Map<number, (options: ScenarioOptions) => Promise<Scenario>>(mapEntries);

if (mapIndexToConstructor.size !== mapEntries.length) {
    throw new Error('Scenario index is not unique');
}

export const scenariosCount = mapIndexToConstructor.size;

export async function createScenarioByCurriculumState(curriculumState: CurriculumState, options: ScenarioOptions): Promise<Scenario> {
    let constructor = createScenarioSoloStatic;

    let weights = [];
    let totalWeight = 0;
    for (let i = 0, minSuccessRatio = 1; i < mapIndexToConstructor.size; i++) {
        let successRatio: number | undefined = curriculumState.mapScenarioIndexToSuccessRatio[i];

        if (successRatio === undefined && minSuccessRatio < 0.65) {
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
                console.warn(`Scenario ${ i } not found, using default scenario static`);
                return createScenarioSoloStatic;
            })();
            break;
        }
        r -= weight;
    }

    return constructor(options);
}
