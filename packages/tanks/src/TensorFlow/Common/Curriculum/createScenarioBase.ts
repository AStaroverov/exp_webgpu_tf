import { createBattlefield } from './createBattlefield.ts';
import { addRandomTanks } from './Utils/addRandomTanks.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';
import { getTankTeamId } from '../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { getSuccessRatio, getTeamHealth } from './utils.ts';
import { Scenario } from './types.ts';
import { max } from '../../../../../../lib/math.ts';
import { createPilotsPlugin } from '../../../Pilots/createPilotsPlugin.ts';
import { CurrentActorAgent } from '../../../Pilots/Agents/CurrentActorAgent.ts';
import { query } from 'bitecs';
import { Tank } from '../../../Game/ECS/Components/Tank.ts';
import { getTeamsCount } from '../../../Game/ECS/Components/TeamRef.ts';

export const indexScenarioWithAlliesStatic = 1;

export async function createScenarioBase(options?: Parameters<typeof createBattlefield>[0] & {
    alliesCount?: number;
    enemiesCount?: number;
}): Promise<Scenario> {
    const game = createBattlefield(options);
    const pilots = createPilotsPlugin(game);

    const alliesCount = options?.alliesCount ?? randomRangeInt(1, 3);
    const enemiesCount = options?.enemiesCount ?? max(1, alliesCount + randomRangeInt(-1, 1));
    const tanks = addRandomTanks([[0, alliesCount], [1, enemiesCount]]);
    const activeTeam = getTankTeamId(tanks[0]);
    const initialTeamHealth = getTeamHealth(tanks);

    pilots.setPilot(tanks[0], new CurrentActorAgent(tanks[0], true), game);
    pilots.toggle(true);

    return {
        ...game,
        ...pilots,

        index: indexScenarioWithAlliesStatic,

        getTankEids: () => {
            return query(game.world, [Tank]);
        },

        getTeamsCount(): number {
            return getTeamsCount();
        },

        getSuccessRatio() {
            return getSuccessRatio(activeTeam, initialTeamHealth, getTeamHealth(tanks));
        },
    };
}