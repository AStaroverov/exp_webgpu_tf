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
    let constructor = createScenario0;

    for (const [index, createScenario] of Object.entries(mapIndexToConstructor)) {
        constructor = createScenario;

        const successRatio = curriculumState.mapScenarioIndexToSuccessRatio[Number(index)] ?? 0;

        if (successRatio < 0.75 || random() < 0.2) {
            break;
        }
    }

    return constructor(createAgent, options);
}