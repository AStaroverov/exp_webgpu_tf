import { createGame } from '../../../Game/createGame.ts';
import { BehaviorSubject } from 'rxjs';
import { GAME_MAP_SIZE } from './def.ts';

type Engine = ReturnType<typeof createGame>;

export const engine$ = new BehaviorSubject<undefined | Engine>(undefined);

export const getEngine = (): Engine => {
    !engine$.value && engine$.next(createGame({ width: GAME_MAP_SIZE, height: GAME_MAP_SIZE, withPlayer: false }));
    return engine$.value as Engine;
};

export const destroyEngine = () => {
    engine$.value?.destroy();
    engine$.next(undefined);
};
