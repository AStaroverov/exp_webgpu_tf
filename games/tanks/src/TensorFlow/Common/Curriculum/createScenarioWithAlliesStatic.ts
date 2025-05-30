import { createBattlefield } from './createBattlefield.ts';
import { CurrentActorAgent, TankAgent } from './Agents/CurrentActorAgent.ts';
import { EntityId } from 'bitecs';
import { addRandomTanks } from './Utils/addRandomTanks.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';
import { getTankTeamId } from '../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { getSuccessRatio, getTeamHealth } from './utils.ts';
import { Scenario } from './types.ts';
import { max } from '../../../../../../lib/math.ts';

export const indexScenarioWithAlliesStatic = 1;

export async function createScenarioWithAlliesStatic(options: Parameters<typeof createBattlefield>[0] & {
    alliesCount?: number;
    enemiesCount?: number;
}): Promise<Scenario> {
    const game = createBattlefield(options);
    const alliesCount = options.alliesCount ?? randomRangeInt(1, 3);
    const enemiesCount = options.enemiesCount ?? max(1, alliesCount + randomRangeInt(-1, 1));
    const tanks = addRandomTanks([[0, alliesCount], [1, enemiesCount]]);
    const activeTeam = getTankTeamId(tanks[0]);
    const initialTeamHealth = getTeamHealth(tanks);

    const mapTankIdToAgent = new Map<number, TankAgent>();
    mapTankIdToAgent.set(tanks[0], new CurrentActorAgent(tanks[0], true));

    await Promise.all(Array.from(mapTankIdToAgent.values()).map(agent => agent.sync?.()));

    return {
        ...game,
        index: indexScenarioWithAlliesStatic,
        destroy: () => {
            game.destroy();
            mapTankIdToAgent.forEach(agent => agent.dispose?.());
        },
        getAliveActors() {
            return game.getTankEids()
                .filter((eid) => mapTankIdToAgent.has(eid) && mapTankIdToAgent.get(eid) instanceof CurrentActorAgent)
                .map((eid) => mapTankIdToAgent.get(eid) as CurrentActorAgent);
        },
        getAliveAgents() {
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
        getAgents() {
            return Array.from(mapTankIdToAgent.values());
        },
        getFreeTankEids() {
            return game.getTankEids().filter((eid) => !mapTankIdToAgent.has(eid));
        },
    };
}