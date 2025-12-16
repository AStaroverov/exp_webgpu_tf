import { BehaviorSubject, map, merge, switchMap, tap } from 'rxjs';
import { min } from '../../../../../../lib/math.ts';
import { frameInterval } from '../../../../../../lib/Rx/frameInterval.ts';
import { getEngine } from './engine.ts';

export const GameState$ = new BehaviorSubject({
    isStarted: false,
});

export const toggleGame = (v?: boolean) => {
    GameState$.next({
        ...GameState$.value,
        isStarted: v ?? !GameState$.value.isStarted,
    });
};

export function GameStateEffects() {
    const gameTicker = GameState$.pipe(
        switchMap(({ isStarted }) => {
            const step = isStarted ? 1 : 6;
            return frameInterval(step * 16).pipe(
                map((dt) => min(16.6667, dt)),
                tap((dt) => getEngine().gameTick(step * dt)),
            );
        }),
    );

    return merge(gameTicker);
}

