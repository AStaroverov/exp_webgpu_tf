import { BehaviorSubject, map, merge, switchMap, tap, timer, filter, take } from 'rxjs';
import { min } from '../../../../../../lib/math.ts';
import { frameInterval } from '../../../../../../lib/Rx/frameInterval.ts';
import { getEngine } from './engine.ts';
import { isPlayerDead, restartBulletHellGame, spawnSingleEnemy } from './gameMethods.ts';

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

    // Spawn enemies every 3 seconds if game is started
    const spawner = BulletHellState$.pipe(
        switchMap(({ isStarted }) => {
            if (!isStarted) return [];
            return timer(1000, 3000).pipe(
                tap(() => spawnSingleEnemy())
            );
        })
    );

    // Check for player death and restart
    const deathChecker = BulletHellState$.pipe(
        switchMap(({ isStarted }) => {
            if (!isStarted) return [];
            return frameInterval(10).pipe( // Check every 10 frames
                filter(() => isPlayerDead()),
                take(1), // Only trigger once per game session
                tap(() => {
                    // Small delay before restart to show death
                    setTimeout(() => restartBulletHellGame(), 1500);
                })
            );
        })
    );

    return merge(gameTicker, spawner, deathChecker);
}
