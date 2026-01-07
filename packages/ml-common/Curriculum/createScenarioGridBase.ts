import { CurrentActorAgent } from '../../tanks/src/Pilots/Agents/CurrentActorAgent.ts';
import { Pilot } from '../../tanks/src/Pilots/Components/Pilot.ts';
import { resetSpawnGrid } from '../../tanks/src/Arena/State/Game/SpawnGrid.ts';
import { createScenarioCore, ScenarioCoreOptions } from './createScenarioCore.ts';
import { Scenario } from './types.ts';
import { addTanksWithGrid } from './Utils/addTanksWithGrid.ts';
import { addFauna } from './Utils/addFauna.ts';
import { fillAlliesWithAgents } from './Utils/fillAlliesWithAgents.ts';

export function createScenarioGridBase(options: ScenarioCoreOptions & {
    alliesCount?: number;
    enemiesCount?: number;
}): Scenario {
    const scenario = createScenarioCore(options);
    const alliesCount = options.alliesCount ?? 3;
    const enemiesCount = options.enemiesCount ?? alliesCount;

    resetSpawnGrid();
    const tanks = addTanksWithGrid([[0, alliesCount], [1, enemiesCount]]);
    addFauna();

    Pilot.addComponent(scenario.world, tanks[0], new CurrentActorAgent(tanks[0], scenario.train));
    fillAlliesWithAgents(scenario);

    return scenario;
}
