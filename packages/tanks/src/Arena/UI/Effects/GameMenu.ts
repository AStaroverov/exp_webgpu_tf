import { GameMenuState$ } from '../../State/GameMenuState.ts';
import { filter, fromEvent, tap } from 'rxjs';

export function GameMenuEffects() {
    return fromEvent<KeyboardEvent>(document, 'keydown').pipe(
        filter(({ key }) => key === 'Escape'),
        tap(() => {
            GameMenuState$.next({
                isOpen: !GameMenuState$.value.isOpen,
            });
        }),
    );
}