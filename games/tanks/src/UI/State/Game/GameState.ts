import { BehaviorSubject, map, merge, switchMap, tap } from 'rxjs';
import { max } from '../../../../../../lib/math.ts';
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
            return frameInterval(step).pipe(
                map((d) => max(step * 16.6667, d)),
                tap((d) => getEngine().gameTick(d)),
            );
        }),
    );

    return merge(gameTicker);
}

