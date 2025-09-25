import { EntityId } from 'bitecs';
import { GameDI } from '../Game/DI/GameDI.ts';
import { SystemGroup } from '../Game/ECS/Plugins/systems.ts';
import { TankAgent } from './Agents/CurrentActorAgent.ts';
import { getAliveActors, getAlivePilots, getFreeTankEids, getPilot, getPilots, Pilot } from './Components/Pilot.ts';
import { PilotsState } from './Singelton/PilotsState.ts';
import { createPilotSystem } from './Systems/createPilotSystem.ts';

export function createPilotsPlugin(game: typeof GameDI) {
    game.plugins.addSystem(SystemGroup.Before, createPilotSystem());
    game.plugins.addDestroy(() => {
        PilotsState.toggle(false);
        Pilot.dispose();
    });

    const setPlayerPilot = (tankEid: number) => {
        // TODO: Move impl to createPilotsManager
        GameDI.enablePlayer();
        GameDI.setPlayerTank(tankEid);
    };

    return {
        ...PilotsState,
        setPlayerPilot,
        setPilot,
        getPilot,
        getPilots,
        getAliveActors,
        getAlivePilots,
        getFreeTankEids,
    };
}

function setPilot(tankEid: EntityId, agent: TankAgent, { world } = GameDI) {
    Pilot.addComponent(world, tankEid, agent);
}
