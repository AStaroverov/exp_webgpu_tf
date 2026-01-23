import { gameState$, } from "../modules/gameState";
import { map, merge, switchMap, tap } from "rxjs";
import { frameInterval } from "../../../../../../lib/Rx/frameInterval";
import { getEngine } from "../engine";
import { min } from "../../../../../../lib/math";

const gameTicker = gameState$.pipe(
    switchMap(({ isStarted }) => {
        const step = isStarted ? 1 : 6;
        return frameInterval(step * 16).pipe(
            map((dt) => min(16.6667, dt)),
            tap((dt) => getEngine().gameTick(step * dt)),
        );
    }),
);

export function initGameInitEffect() {
    // Enable player on first load
    getEngine().enablePlayer();
    
    return merge(
        gameTicker,
    );
}