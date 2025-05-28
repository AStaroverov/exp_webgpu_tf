import { CSSProperties, useCallback } from 'react';
import { GameMenuState$, toggleGameMenu } from '../State/GameMenuState.ts';
import { useObservable } from '../../../../../lib/React/useSyncObservable.ts';
import { exitGame } from '../State/Game/playerMethods.ts';
import { Button } from '../Components/Button.tsx';
import { Card } from '../Components/Card.tsx';

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
        <Card
            className={ `${ className } p-2 gap-2 absolute transition-all top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2` }
            style={ style }
        >
            <div>Menu</div>

            <Button color="warning" onClick={ handleExit }>Exit from battle</Button>

            <Button color="primary" onClick={ handleResume }>Resume</Button>
        </Card>
    );
}


