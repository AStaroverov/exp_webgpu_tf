import { CSSProperties, useEffect, useRef } from 'react';
import { GarageScreen } from './GarageScreen.tsx';
import { EscMenu } from './EscMenu.tsx';
import { useObservable } from '../../../../../../lib/React/useSyncObservable.ts';
import { GameState$ } from '../../State/Game/GameState.ts';
import { setRenderTarget } from '../../State/Game/RenderTarget.ts';

import { GAME_MAP_SIZE } from '../../State/Game/def.ts';

export function BaseScreen({ className, style }: { className?: string, style?: CSSProperties }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const canvasScale = Math.min(window.innerWidth / (canvasRef.current?.offsetWidth ?? 1), window.innerHeight / (canvasRef.current?.offsetHeight ?? 1));
    const { isStarted } = useObservable(GameState$, GameState$.value);

    useEffect(() => {
        setRenderTarget(canvasRef.current);
        return () => setRenderTarget(null);
    }, [canvasRef.current]);

    return (
        <div className={ `${ className } flex items-center justify-center` } style={ style }>
            <canvas
                ref={ canvasRef }
                className={ `absolute transition-all origin-left
                    ${ isStarted ? 'left-0' : 'left-20' }
                ` }
                style={ {
                    width: `${ GAME_MAP_SIZE }px`,
                    height: `${ GAME_MAP_SIZE }px`,
                    transform: `scale(${ isStarted ? canvasScale : '50%' })`,
                } }
            />
            <GarageScreen className={ `absolute transition-all ${ isStarted ? 'left-0 opacity-0 ' : 'right-60' }` }/>
            <EscMenu/>
        </div>
    );
}
