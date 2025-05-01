import { TankAgent } from '../../PPO/Actor/ActorAgent.ts';
import { EntityId } from 'bitecs';
import { Scenario } from './types.ts';
import { CurriculumState } from '../../PPO/channels.ts';
import { createScenario0 } from './createScenario0.ts';
import { random } from '../../../../../../lib/random.ts';
import { createScenario1 } from './createScenario1.ts';
import { createBattlefield } from './createBattlefield.ts';

type ScenarioConstructor = (
    createAgent: (tankEid: EntityId) => TankAgent,
    options: Parameters<typeof createBattlefield>[0],
) => Promise<Scenario>;

const mapIndexToConstructor: Record<number, ScenarioConstructor> = {
    0: createScenario0,
    1: createScenario1,
};

export async function createScenarioByCurriculumState(
    curriculumState: CurriculumState,
    createAgent: (tankEid: EntityId) => TankAgent,
    options: Parameters<typeof createBattlefield>[0],
): Promise<Scenario> {

    const entries = Object
        .entries(curriculumState.mapScenarioIndexToSuccessRatio)
        .sort((a, b) => Number(b[0]) - Number(a[0]));

    let constructor = createScenario0;

    for (const [index, successRatio] of entries) {
        constructor = mapIndexToConstructor[Number(index)] || constructor;

        if (successRatio < 0.8 || random() < 0.2) {
            break;
        }
    }

    return constructor(createAgent, options);
}