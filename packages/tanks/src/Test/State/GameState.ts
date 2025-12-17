import { BehaviorSubject, map, merge, switchMap, tap, filter, take } from 'rxjs';
import { min } from '../../../../../lib/math.ts';
import { frameInterval } from '../../../../../lib/Rx/frameInterval.ts';
import { getEngine } from './engine.ts';
import { isPlayerDead, restartTestGame } from './gameMethods.ts';

export const TestGameState$ = new BehaviorSubject({
    isStarted: false,
    enemyCount: 0,
});

export const toggleTestGame = (v?: boolean) => {
    TestGameState$.next({
        ...TestGameState$.value,
        isStarted: v ?? !TestGameState$.value.isStarted,
    });
};

export const incrementEnemyCount = () => {
    TestGameState$.next({
        ...TestGameState$.value,
        enemyCount: TestGameState$.value.enemyCount + 1,
    });
};

export function TestGameStateEffects() {
    const gameTicker = TestGameState$.pipe(
        switchMap(({ isStarted }) => {
            const step = isStarted ? 1 : 6;
            return frameInterval(step * 16).pipe(
                map((dt) => min(16.6667, dt)),
                tap((dt) => getEngine().gameTick(step * dt)),
            );
        }),
    );

    // Check for player death and restart
    const deathChecker = TestGameState$.pipe(
        switchMap(({ isStarted }) => {
            if (!isStarted) return [];
            return frameInterval(10).pipe( // Check every 10 frames
                filter(() => isPlayerDead()),
                take(1), // Only trigger once per game session
                tap(() => {
                    // Small delay before restart to show death
                    setTimeout(() => restartTestGame(), 1500);
                })
            );
        })
    );

    return merge(gameTicker, deathChecker);
}

