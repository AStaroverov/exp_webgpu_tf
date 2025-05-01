import { createBattlefield } from './createBattlefield.ts';
import { TankAgent } from '../../PPO/Actor/ActorAgent.ts';
import { EntityId } from 'bitecs';
import { addRandomTanks } from './addRandomTanks.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';
import { getTankTeamId } from '../../../ECS/Entities/Tank/TankUtils.ts';
import { getSuccessRatio, getTeamHealth } from './utils.ts';
import { Scenario } from './types.ts';

export async function createScenario0(
    createAgent: (tankEid: EntityId) => TankAgent,
    options: Parameters<typeof createBattlefield>[0],
): Promise<Scenario> {
    const game = await createBattlefield(options);
    const tanks = addRandomTanks([[0, randomRangeInt(1, 3)], [1, randomRangeInt(1, 3)]]);
    const activeTeam = getTankTeamId(tanks[0]);
    const initialTeamHealth = getTeamHealth(tanks);

    const mapTankIdToAgent = new Map<number, TankAgent>();
    mapTankIdToAgent.set(tanks[0], createAgent(tanks[0]));

    await Promise.all(Array.from(mapTankIdToAgent.values()).map(agent => agent.sync()));

    return {
        ...game,
        index: 0,
        destroy: () => {
            game.destroy();
            mapTankIdToAgent.forEach(agent => agent.dispose());
        },
        getAgents() {
            return game.getTankEids()
                .filter((eid) => mapTankIdToAgent.has(eid))
                .map((eid) => mapTankIdToAgent.get(eid) as TankAgent);
        },
        getSuccessRatio() {
            return getSuccessRatio(activeTeam, initialTeamHealth, getTeamHealth(tanks));
        },

        addAgent(tankEid: EntityId, agent: TankAgent) {
            mapTankIdToAgent.set(tankEid, agent);
        },
        getAgent(tankEid: EntityId) {
            return mapTankIdToAgent.get(tankEid);
        },
    };
}