import { Scenario } from './types.ts';
import { CurriculumState } from '../../PPO/channels.ts';
import { createScenarioStatic, indexScenarioStatic } from './createScenarioStatic.ts';
import { random } from '../../../../../../lib/random.ts';
import { createScenarioStaticWithCoop, indexScenarioStaticWithCoop } from './createScenarioStaticWithCoop.ts';
import { createBattlefield } from './createBattlefield.ts';
import { createScenarioWithMovingAgents, indexScenarioWithMovingAgents } from './createScenarioWithMovingAgents.ts';

type ScenarioOptions = Parameters<typeof createBattlefield>[0];

const mapIndexToConstructor = new Map([
    [indexScenarioStatic, createScenarioStatic],
    [indexScenarioStaticWithCoop, createScenarioStaticWithCoop],
    [indexScenarioWithMovingAgents, createScenarioWithMovingAgents],
]);

if (mapIndexToConstructor.size !== 3) {
    throw new Error('Scenario index is not unique');
}

export async function createScenarioByCurriculumState(curriculumState: CurriculumState, options: ScenarioOptions): Promise<Scenario> {
    let constructor = createScenarioStatic;

    for (const [index, createScenario] of Object.entries(mapIndexToConstructor)) {
        constructor = createScenario;

        const successRatio = curriculumState.mapScenarioIndexToSuccessRatio[Number(index)] ?? 0;

        if (successRatio < 0.75 || random() < 0.2) {
            break;
        }
    }

    return constructor(options);
}