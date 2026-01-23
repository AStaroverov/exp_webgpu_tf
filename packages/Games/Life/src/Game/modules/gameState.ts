import { BehaviorSubject } from 'rxjs';

export const gameState$ = new BehaviorSubject({
    isStarted: true,  // Auto-start for Game of Life
});

export const toggleGame = (v?: boolean) => {
    gameState$.next({
        ...gameState$.value,
        isStarted: v ?? !gameState$.value.isStarted,
    });
};
