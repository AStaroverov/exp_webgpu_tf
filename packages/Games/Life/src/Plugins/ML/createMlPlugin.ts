import { GameDI } from '../../GameEngine/DI/GameDI.js';
import { SystemGroup } from '../../GameEngine/ECS/Plugins/systems.js';
import { createMlScoreSystem } from './createMlScoreSystem.js';
import { MLState } from './MlState.js';

export function createMLPlugin(game: typeof GameDI) {
    const mlScoreSystem = createMlScoreSystem();
    game.plugins.addSystem(SystemGroup.Before, mlScoreSystem.tick);
    game.plugins.addDestroy(mlScoreSystem.dispose);
    return MLState;
}
