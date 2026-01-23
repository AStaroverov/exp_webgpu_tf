import { GameDI } from '../../GameEngine/DI/GameDI.js';
import { SystemGroup } from '../../GameEngine/ECS/Plugins/systems.js';
import { Pilot } from './Components/Pilot.js';
import { PilotsState } from './Singelton/PilotsState.js';
import { createPilotSystem } from './Systems/createPilotSystem.js';

export function createPilotsPlugin(game: typeof GameDI) {
    game.plugins.addSystem(SystemGroup.Before, createPilotSystem());
    game.plugins.addDestroy(() => {
        PilotsState.toggle(false);
        Pilot.dispose();
    });

    return PilotsState;
}
