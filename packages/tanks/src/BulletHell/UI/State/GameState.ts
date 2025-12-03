import { BehaviorSubject, map, merge, switchMap, tap } from 'rxjs';
import { min } from '../../../../../../lib/math.ts';
import { frameInterval } from '../../../../../../lib/Rx/frameInterval.ts';
import { getEngine } from './engine.ts';

export const BulletHellState$ = new BehaviorSubject({
    isStarted: false,
});

export const toggleBulletHellGame = (v?: boolean) => {
    BulletHellState$.next({
        ...BulletHellState$.value,
        isStarted: v ?? !BulletHellState$.value.isStarted,
    });
};

export function BulletHellStateEffects() {
    const gameTicker = BulletHellState$.pipe(
        switchMap(({ isStarted }) => {
            const step = isStarted ? 1 : 6;
            return frameInterval(step).pipe(
                map((dt) => min(16.6667, dt)),
                tap((dt) => getEngine().gameTick(step * dt)),
            );
        }),
    );

    return merge(gameTicker);
}
