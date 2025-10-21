import { addComponent, EntityId, Not, query, World } from 'bitecs';
import { isFunction } from 'lodash-es';
import { GameDI } from '../../Game/DI/GameDI.ts';
import { Tank } from '../../Game/ECS/Components/Tank.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';
import { CurrentActorAgent, TankAgent } from '../Agents/CurrentActorAgent.ts';

export const Pilot = {
    agent: {} as Record<EntityId, TankAgent>,

    dispose() {
        getPilots().forEach(agent => agent?.dispose?.());
        Pilot.agent = {};
    },

    addComponent(world: World, eid: EntityId, agent: TankAgent) {
        addComponent(world, eid, Pilot);
        Pilot.agent[eid] = agent;
    },

    isSynced() {
        return getPilots().every(pilot => isFunction(pilot.isSynced) ? pilot.isSynced() : true);
    },
};

export function getPilot(tankEid: EntityId) {
    return Pilot.agent[tankEid];
}

export function getAliveActors({ world } = GameDI) {
    return query(world, [Pilot])
        .map((eid) => Pilot.agent[eid])
        .filter((agent) => agent instanceof CurrentActorAgent && getTankHealth(agent.tankEid) > 0);
}

export function getAlivePilots({ world } = GameDI) {
    return query(world, [Pilot])
        .map((eid) => Pilot.agent[eid])
        .filter((agent) => agent != null && getTankHealth(agent.tankEid) > 0);
}

export function getPilots(): readonly TankAgent[] {
    return Object.values(Pilot.agent);
}

export function getFreeTankEids({ world } = GameDI) {
    return query(world, [Tank, Not(Pilot)]);
}
