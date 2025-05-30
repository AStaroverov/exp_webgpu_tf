import {
    getAliveActors,
    getAlivePilots,
    getFreeTankEids,
    getPilot,
    getPilots,
    getPilotType,
    Pilot,
    PilotType,
} from './Components/Pilot.ts';
import { GameDI } from '../Game/DI/GameDI.ts';
import { SystemGroup } from '../Game/ECS/Plugins/systems.ts';
import { createPilotSystem } from './Systems/createPilotSystem.ts';
import { PilotsState } from './Singelton/PilotsState.ts';
import { EntityId, hasComponent, removeComponent } from 'bitecs';
import { TankInputTensor } from './Components/TankState.ts';
import { ValueOf } from '../../../../lib/Types';
import { createInitPilotSystem } from './Systems/createInitPilotSystem.ts';

export function createPilotsPlugin(game: typeof GameDI) {
    game.plugins.addSystem(SystemGroup.Before, createInitPilotSystem());
    game.plugins.addSystem(SystemGroup.Before, createPilotSystem());

    game.plugins.addDestroy(() => {
        PilotsState.toggle(false);
        Pilot.dispose();
    });

    return {
        ...PilotsState,
        setPlayerPilot,
        removePilot,
        setPilot,
        getPilot,
        getPilotType,
        getPilots,
        getAliveActors,
        getAlivePilots,
        getFreeTankEids,
    };
}

function setPlayerPilot(tankEid: number, { world } = GameDI) {
    removePilot(tankEid);
    Pilot.addComponent(world, tankEid, PilotType.Player);
}

function setPilot(tankEid: EntityId, agentType: ValueOf<typeof PilotType>, { world } = GameDI) {
    removePilot(tankEid);

    TankInputTensor.addComponent(world, tankEid);
    Pilot.addComponent(world, tankEid, agentType);
}

function removePilot(tankEid: EntityId, { world } = GameDI) {
    hasComponent(world, tankEid, Pilot) && removeComponent(world, tankEid, TankInputTensor);
    Pilot.removeComponent(world, tankEid);
}
