import { createGame } from '../../Game/createGame.ts';
import { BehaviorSubject } from 'rxjs';
import { createPilotsPlugin } from '../../Pilots/createPilotsPlugin.ts';
import { createPlayer } from '../../Game/ECS/Entities/Player.ts';

type Engine = ReturnType<typeof createGame> & {
    pilots: ReturnType<typeof createPilotsPlugin>
};

export const engine$ = new BehaviorSubject<undefined | Engine>(undefined);

export function getEngine(): Engine {
    if (!engine$.value) {
        const core = createGame({ width: window.innerWidth, height: window.innerHeight });
        const realPlayerId = createPlayer(0);
        core.setPlayerId(realPlayerId);
        engine$.next({
            pilots: createPilotsPlugin(core),
            ...core,
        });
    }

    return engine$.value as Engine;
};

export function destroyEngine() {
    engine$.value?.destroy();
    engine$.next(undefined);
};
