import { gameState$, } from "../modules/gameState";
import { clearSlots } from "../modules/lobbySlots";
import { distinctUntilChanged, filter, map, merge, switchMap, tap } from "rxjs";
import { frameInterval } from "../../../../../../../lib/Rx/frameInterval";
import { getEngine } from "../engine";
import { min } from "../../../../../../../lib/math";
import { addFauna, addHarvester } from "../gameMethods";

const gameStoped$ = gameState$.pipe(
    distinctUntilChanged((a, b) => a.isStarted === b.isStarted),
    filter(({ isStarted }) => !isStarted),
);

const gameStarted$ = gameState$.pipe(
    distinctUntilChanged((a, b) => a.isStarted === b.isStarted),
    filter(({ isStarted }) => isStarted),
);

const gameTicker = gameState$.pipe(
    switchMap(({ isStarted }) => {
        const step = isStarted ? 1 : 6;
        return frameInterval(step * 16).pipe(
            map((dt) => min(16.6667, dt)),
            tap((dt) => getEngine().gameTick(step * dt)),
        );
    }),
);

const resetSlotsOnInit$ = gameStoped$.pipe(
    tap(() => {
        clearSlots();
    })
);

const addFaunaOnStart$ = gameStarted$.pipe(
    tap(() => {
        addFauna();
    })
);

const fillPlayerSlotsOnInit$ = gameStoped$.pipe(
    tap(() => {
        const engine = getEngine();
        const harvesterEid = addHarvester()
        engine.setPlayerVehicle(harvesterEid);
        engine.setCameraTarget(harvesterEid);
    }),
);

const enablePlayerOnStart$ = gameStarted$.pipe(
    tap(() => {
        getEngine().enablePlayer();
    })
);

export function initGameInitEffect() {
    return merge(
        gameTicker,
        addFaunaOnStart$,
        resetSlotsOnInit$,
        fillPlayerSlotsOnInit$,
        enablePlayerOnStart$,
    );
}