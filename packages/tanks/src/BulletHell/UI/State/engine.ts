import { createGame } from '../../../Game/createGame.ts';
import { BehaviorSubject } from 'rxjs';
import { BULLET_HELL_MAP_SIZE } from './def.ts';
import { createPilotsPlugin } from '../../../Pilots/createPilotsPlugin.ts';

type Engine = ReturnType<typeof createGame> & {
    pilots: ReturnType<typeof createPilotsPlugin>
};

export const engine$ = new BehaviorSubject<undefined | Engine>(undefined);

export const getEngine = (): Engine => {
    if (!engine$.value) {
        const core = createGame({ width: BULLET_HELL_MAP_SIZE, height: BULLET_HELL_MAP_SIZE });
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
