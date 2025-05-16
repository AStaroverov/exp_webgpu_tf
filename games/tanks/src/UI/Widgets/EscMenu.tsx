import { CSSProperties, useCallback } from 'react';
import { GameMenuState$, toggleGameMenu } from '../State/GameMenuState.ts';
import { useObservable } from '../../../../../lib/React/useSyncObservable.ts';
import { exitGame } from '../State/Game/playerMethods.ts';

export function EscMenu({ className, style }: {
    className?: string,
    style?: CSSProperties,
}) {
    const { isOpen } = useObservable(GameMenuState$, GameMenuState$.value);
    const handleExit = useCallback(() => {
        exitGame();
        toggleGameMenu(false);
    }, []);
    const handleResume = useCallback(() => {
        toggleGameMenu(false);
    }, []);

    return !isOpen ? null : (
        <div
            className={ `${ className } absolute transition-all top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2` }
            style={ style }
        >
            <div>Menu</div>

            <div onClick={ handleExit }>Exit from battle</div>

            <div onClick={ handleResume }>Resume</div>
        </div>
    );
}


