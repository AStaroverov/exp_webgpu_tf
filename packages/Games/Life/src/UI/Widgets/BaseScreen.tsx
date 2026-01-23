import { CSSProperties, useEffect, useRef } from 'react';
import { GAME_MAP_SIZE } from '../../Game/def';
import { gameState$ } from '../../Game/modules/gameState';
import { setRenderTarget } from '../../Game/RenderTarget';
import { useObservable } from '../../../../../../lib/React/useSyncObservable';

export function BaseScreen({ className, style }: { className?: string, style?: CSSProperties }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    // const canvasScale = Math.min(window.innerWidth / (canvasRef.current?.offsetWidth ?? 1), window.innerHeight / (canvasRef.current?.offsetHeight ?? 1));
    const { isStarted } = useObservable(gameState$, gameState$.value);

    useEffect(() => {
        setRenderTarget(canvasRef.current);
        return () => setRenderTarget(null);
    }, [canvasRef.current]);

    return (
        <div className={ `${ className } flex items-center justify-center` } style={ style }>
            <canvas
                ref={ canvasRef }
                className={ `absolute transition-all ${ isStarted ? 'inset-0' : 'origin-left left-20' }` }
                style={ isStarted 
                    ? { width: '100%', height: '100%' }
                    : {
                        width: `${ GAME_MAP_SIZE }px`,
                        height: `${ GAME_MAP_SIZE }px`,
                        transform: 'scale(50%)',
                    }
                }
            />
        </div>
    );
}
