import { addComponent, EntityId, hasComponent, Not, query, removeComponent, World } from 'bitecs';
import { CurrentActorAgent, TankAgent } from '../Agents/CurrentActorAgent.ts';
import { GameDI } from '../../Game/DI/GameDI.ts';
import { Tank } from '../../Game/ECS/Components/Tank.ts';
import { ValueOf } from '../../../../../lib/Types';

export const PilotType = Object.freeze({
    None: 0,
    Player: 1,

    AgentLearnable: 2,
    Agent: 3,
    AgentRandom: 4,

    // postfix for assets
    Agent31: 31,
    Agent32: 32,

    BotOnlyMoving: 201,
    BotOnlyShooting: 202,
    BotSimple: 203,
    BotStrong: 204,
});

export const PilotAgents = new Map<EntityId, undefined | TankAgent>();

export const Pilot = {
    type: [] as ValueOf<typeof PilotType>[],

    dispose() {
        getPilots().forEach(agent => agent?.dispose?.());
    },

    addComponent(world: World, eid: EntityId, type: ValueOf<typeof PilotType>) {
        addComponent(world, eid, Pilot);
        Pilot.type[eid] = type;
    },

    removeComponent(world: World, eid: EntityId) {
        hasComponent(world, eid, Pilot) && removeComponent(world, eid, Pilot);
        Pilot.type[eid] = PilotType.None;
        PilotAgents.get(eid)?.dispose?.();
        PilotAgents.delete(eid);
    },
};

export function getPilot(tankEid: EntityId) {
    return PilotAgents.get(tankEid);
}

export function getPilotType(tankEid: EntityId) {
    return Pilot.type[tankEid];
}

export function getAliveActors({ world } = GameDI) {
    return query(world, [Pilot])
        .map((eid) => PilotAgents.get(eid))
        .filter((agent) => agent instanceof CurrentActorAgent);
}

export function getAlivePilots({ world } = GameDI) {
    return query(world, [Pilot])
        .map((eid) => PilotAgents.get(eid))
        .filter((agent) => agent != null);
}

export function getPilots(): readonly TankAgent[] {
    return Array.from(PilotAgents.values()).filter(agent => agent != null);
}

export function getFreeTankEids({ world } = GameDI) {
    return query(world, [Tank, Not(Pilot)]);
}
