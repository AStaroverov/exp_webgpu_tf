import { query } from 'bitecs';
import { randomRangeInt } from '../../../lib/random.ts';
import { Vehicle } from '../../tanks/src/Game/ECS/Components/Vehicle.ts';
import { getTankTeamId } from '../../tanks/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { CurrentActorAgent } from '../../tanks/src/Pilots/Agents/CurrentActorAgent.ts';
import { createPilotsPlugin } from '../../tanks/src/Pilots/createPilotsPlugin.ts';
import { createBattlefield } from './createBattlefield.ts';
import { Scenario } from './types.ts';
import { getSuccessRatio as computeSuccessRatio, getTeamHealth } from './utils.ts';
import { addRandomTanks } from './Utils/addRandomTanks.ts';
import { fillAlliesWithAgents } from './Utils/fillAlliesWithAgents.ts';
import { getTeamsCount } from '../../tanks/src/Game/ECS/Components/TeamRef.ts';

export const indexScenarioWithAlliesStatic = 1;

export function createScenarioBase(options: Parameters<typeof createBattlefield>[0] & {
    index: number;
    train?: boolean;
    alliesCount?: number;
    enemiesCount?: number;
}): Scenario {
    const game = createBattlefield(options);
    const pilots = createPilotsPlugin(game);

    const isTrain = options.train ?? true;
    const alliesCount = options.alliesCount ?? randomRangeInt(1, 3);
    const enemiesCount = options.enemiesCount ?? alliesCount;
    const tanks = addRandomTanks([[0, alliesCount], [1, enemiesCount]]);
    const activeTeam = getTankTeamId(tanks[0]);
    const initialTeamHealth = getTeamHealth(tanks);

    const getVehicleEids = () => query(game.world, [Vehicle]);
    const getSuccessRatio = () => computeSuccessRatio(activeTeam, initialTeamHealth, getTeamHealth(tanks));
    
    const scenario: Scenario = {
        ...game,
        ...pilots,

        index: options.index,
        isTrain,

        getVehicleEids,
        getTeamsCount,
        getSuccessRatio,
    }

    pilots.setPilot(tanks[0], new CurrentActorAgent(tanks[0], isTrain), game);
    pilots.toggle(true);
    
    fillAlliesWithAgents(scenario, isTrain);

    return scenario;
}