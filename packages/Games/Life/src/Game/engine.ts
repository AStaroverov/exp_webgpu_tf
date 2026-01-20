import { BehaviorSubject } from 'rxjs';
import { GAME_MAP_SIZE } from './def.js';
import { createGame } from '../GameEngine/createGame.js';
import { createPilotsPlugin } from '../Plugins/Pilots/createPilotsPlugin.js';

type Engine = ReturnType<typeof createGame> & {
    pilots: ReturnType<typeof createPilotsPlugin>
};

export const engine$ = new BehaviorSubject<undefined | Engine>(undefined);

export const getEngine = (): Engine => {
    if (!engine$.value) {
        const core = createGame({ cells: GAME_MAP_SIZE, rows: GAME_MAP_SIZE });
        engine$.next({
            pilots: createPilotsPlugin(core),
            ...core,
        });
    }

    return engine$.value as Engine;
};

export const destroyEngine = () => {
    engine$.value?.destroy();
    engine$.next(undefined);
};
