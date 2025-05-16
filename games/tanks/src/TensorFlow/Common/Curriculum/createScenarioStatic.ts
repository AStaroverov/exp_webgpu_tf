import { createBattlefield } from './createBattlefield.ts';
import { CurrentActorAgent, TankAgent } from './Agents/CurrentActorAgent.ts';
import { EntityId } from 'bitecs';
import { addRandomTanks } from './Utils/addRandomTanks.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';
import { getTankTeamId } from '../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { getScenarioIndex, getSuccessRatio, getTeamHealth } from './utils.ts';
import { Scenario } from './types.ts';
import { max } from '../../../../../../lib/math.ts';

export const indexScenarioStatic = getScenarioIndex();

export async function createScenarioStatic(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const game = await createBattlefield(options);
    const count = randomRangeInt(1, 3);
    const tanks = addRandomTanks([[0, count], [1, max(1, count + randomRangeInt(-1, 1))]]);
    const activeTeam = getTankTeamId(tanks[0]);
    const initialTeamHealth = getTeamHealth(tanks);

    const mapTankIdToAgent = new Map<number, TankAgent>();
    mapTankIdToAgent.set(tanks[0], new CurrentActorAgent(tanks[0], true));

    await Promise.all(Array.from(mapTankIdToAgent.values()).map(agent => agent.sync?.()));

    return {
        ...game,
        index: indexScenarioStatic,
        destroy: () => {
            game.destroy();
            mapTankIdToAgent.forEach(agent => agent.dispose?.());
        },
        getActors() {
            return game.getTankEids()
                .filter((eid) => mapTankIdToAgent.has(eid) && mapTankIdToAgent.get(eid) instanceof CurrentActorAgent)
                .map((eid) => mapTankIdToAgent.get(eid) as CurrentActorAgent);
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
        getFreeTankEids() {
            return game.getTankEids().filter((eid) => !mapTankIdToAgent.has(eid));
        },
    };
}