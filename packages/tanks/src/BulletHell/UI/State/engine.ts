import { createGame } from '../../../Game/createGame.ts';
import { BehaviorSubject } from 'rxjs';
import { createPilotsPlugin } from '../../../Pilots/createPilotsPlugin.ts';

type Engine = ReturnType<typeof createGame> & {
    pilots: ReturnType<typeof createPilotsPlugin>
};

export const engine$ = new BehaviorSubject<undefined | Engine>(undefined);

export const getEngine = (): Engine => {
    if (!engine$.value) {
        // Use window size for the game zone (for infinite map, this defines the visible area)
        const core = createGame({ width: window.innerWidth, height: window.innerHeight });
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
