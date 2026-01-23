import { BehaviorSubject } from 'rxjs';
import { GAME_MAP_SIZE } from './def.js';
import { createGame } from '../GameEngine/createGame.js';

type Engine = ReturnType<typeof createGame>;

export const engine$ = new BehaviorSubject<undefined | Engine>(undefined);

export const getEngine = (): Engine => {
    if (!engine$.value) {
        const core = createGame({ cells: GAME_MAP_SIZE, rows: GAME_MAP_SIZE });
        engine$.next(core);
    }

    return engine$.value as Engine;
};

export const destroyEngine = () => {
    engine$.value?.destroy();
    engine$.next(undefined);
};
