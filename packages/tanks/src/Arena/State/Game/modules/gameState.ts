import { BehaviorSubject } from 'rxjs';

export const gameState$ = new BehaviorSubject({
    isStarted: false,
});

export const toggleGame = (v?: boolean) => {
    gameState$.next({
        ...gameState$.value,
        isStarted: v ?? !gameState$.value.isStarted,
    });
};
