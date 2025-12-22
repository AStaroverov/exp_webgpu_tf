import { GameDI } from '../Game/DI/GameDI.ts';
import { SystemGroup } from '../Game/ECS/Plugins/systems.ts';
import { Pilot } from './Components/Pilot.ts';
import { PilotsState } from './Singelton/PilotsState.ts';
import { createPilotSystem } from './Systems/createPilotSystem.ts';

export function createPilotsPlugin(game: typeof GameDI) {
    game.plugins.addSystem(SystemGroup.Before, createPilotSystem());
    game.plugins.addDestroy(() => {
        PilotsState.toggle(false);
        Pilot.dispose();
    });

    return PilotsState;
}
