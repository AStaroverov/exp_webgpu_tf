import { addComponent, EntityId, Not, query, World } from 'bitecs';
import { CurrentActorAgent, TankAgent } from '../Agents/CurrentActorAgent.ts';
import { GameDI } from '../../Game/DI/GameDI.ts';
import { Tank } from '../../Game/ECS/Components/Tank.ts';
import { isFunction } from 'lodash-es';

export const Pilot = {
    agent: [] as TankAgent[],

    sincCount: 0,

    dispose() {
        getPilots().forEach(agent => agent?.dispose?.());
        Pilot.sincCount = 0;
        Pilot.agent.length = 0;
    },

    addComponent(world: World, eid: EntityId, agent: TankAgent) {
        addComponent(world, eid, Pilot);

        Pilot.agent[eid] = agent;

        if (isFunction(agent.sync)) {
            Pilot.sincCount++;
            agent.sync().finally(() => Pilot.sincCount--);
        }
    },

    isSynced() {
        return Pilot.sincCount === 0;
    },
};

export function getPilot(tankEid: EntityId) {
    return Pilot.agent[tankEid];
}

export function getAliveActors({ world } = GameDI) {
    return query(world, [Pilot])
        .map((eid) => Pilot.agent[eid])
        .filter((agent) => agent instanceof CurrentActorAgent);
}

export function getAlivePilots({ world } = GameDI) {
    return query(world, [Pilot])
        .map((eid) => Pilot.agent[eid])
        .filter((agent) => agent != null);
}

export function getPilots(): readonly TankAgent[] {
    return Pilot.agent;
}

export function getFreeTankEids({ world } = GameDI) {
    return query(world, [Tank, Not(Pilot)]);
}
