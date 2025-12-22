import { addComponent, EntityId, Not, query, World } from 'bitecs';
import { GameDI } from '../../Game/DI/GameDI.ts';
import { Vehicle } from '../../Game/ECS/Components/Vehicle.ts';
import { getTankHealth } from '../../Game/ECS/Entities/Tank/TankUtils.ts';
import { CurrentActorAgent, TankAgent } from '../Agents/CurrentActorAgent.ts';

export const Pilot = {
    agent: new Map<EntityId, TankAgent>(),

    dispose() {
        getPilotAgents().forEach(agent => agent?.dispose?.());
        Pilot.agent.clear();
    },

    addComponent(world: World, eid: EntityId, agent: TankAgent) {
        addComponent(world, eid, Pilot);
        Pilot.agent.set(eid, agent);
    },

    getAgent(eid: EntityId) {
        return Pilot.agent.get(eid);
    },

    disposeAgent(eid: EntityId) {
        Pilot.agent.get(eid)?.dispose?.();
        Pilot.agent.delete(eid);
    },
};

export function getPilotAgents({ world } = GameDI) {
    return query(world, [Pilot]).map((eid) => Pilot.agent.get(eid)!);
}

export function getAliveActors({ world } = GameDI) {
    return query(world, [Pilot])
        .map((eid) => Pilot.agent.get(eid))
        .filter((agent): agent is CurrentActorAgent => agent instanceof CurrentActorAgent && getTankHealth(agent.tankEid) > 0);
}

export function getAlivePilots() {
    return getPilotAgents()
        .filter((agent): agent is TankAgent => agent != null && getTankHealth(agent.tankEid) > 0);
}

export function getFreeVehicaleEids({ world } = GameDI) {
    return query(world, [Vehicle, Not(Pilot)]);
}
