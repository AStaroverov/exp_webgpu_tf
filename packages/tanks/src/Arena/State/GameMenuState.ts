import { BehaviorSubject } from 'rxjs';

export const GameMenuState$ = new BehaviorSubject({
    isOpen: false,
});

export function toggleGameMenu(v?: boolean) {
    GameMenuState$.next({
        ...GameMenuState$.value,
        isOpen: v ?? !GameMenuState$.value.isOpen,
    });
}
