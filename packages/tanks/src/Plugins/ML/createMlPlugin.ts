import { GameDI } from '../../Game/DI/GameDI.ts';
import { SystemGroup } from '../../Game/ECS/Plugins/systems.ts';
import { createMlScoreSystem } from './createMlScoreSystem.ts';
import { MLState } from './MlState.ts';

export function createMLPlugin(game: typeof GameDI) {
    const mlScoreSystem = createMlScoreSystem();
    game.plugins.addSystem(SystemGroup.Before, mlScoreSystem.tick);
    game.plugins.addDestroy(mlScoreSystem.dispose);
    return MLState;
}
