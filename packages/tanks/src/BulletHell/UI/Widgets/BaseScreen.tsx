import { CSSProperties, useCallback, useEffect, useRef, useState } from 'react';
import { useObservable } from '../../../../../../lib/React/useSyncObservable.ts';
import { BulletHellState$, toggleBulletHellGame } from '../State/GameState.ts';
import { startBulletHellGame } from '../State/gameMethods.ts';
import { setRenderTarget } from '../State/RenderTarget.ts';
import { BULLET_HELL_MAP_SIZE } from '../State/def.ts';
import { Button } from '../../../UI/Components/Button.tsx';
import { Card } from '../../../UI/Components/Card.tsx';

export function BaseScreen({ className, style }: { className?: string, style?: CSSProperties }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isLoading, setIsLoading] = useState(false);
    const canvasScale = Math.min(
        window.innerWidth / (canvasRef.current?.offsetWidth ?? 1),
        window.innerHeight / (canvasRef.current?.offsetHeight ?? 1)
    );
    const { isStarted } = useObservable(BulletHellState$, BulletHellState$.value);

    useEffect(() => {
        setRenderTarget(canvasRef.current);
        return () => setRenderTarget(null);
    }, [canvasRef.current]);

    const handleStart = useCallback(async () => {
        setIsLoading(true);
        await startBulletHellGame();
        toggleBulletHellGame(true);
        setIsLoading(false);
    }, []);

    const handleStop = useCallback(() => {
        toggleBulletHellGame(false);
    }, []);

    return (
        <div className={`${className} flex items-center justify-center relative`} style={style}>
            <canvas
                ref={canvasRef}
                className="absolute"
                style={{
                    width: `${BULLET_HELL_MAP_SIZE}px`,
                    height: `${BULLET_HELL_MAP_SIZE}px`,
                    transform: `scale(${canvasScale * 0.9})`,
                }}
            />

            {/* Controls */}
            {isStarted && (
                <div className="absolute top-4 left-4 flex gap-4 text-white text-xl font-bold bg-black/50 px-4 py-2 rounded">
                    <Button color="danger" size="sm" onClick={handleStop}>
                        Stop
                    </Button>
                </div>
            )}

            {/* Start Screen */}
            {!isStarted && (
                <Card className="absolute p-8 flex flex-col items-center gap-4 bg-gradient-to-br from-purple-900 to-indigo-900">
                    <h1 className="text-4xl font-bold text-white">Bullet Hell</h1>
                    <p className="text-gray-300">Watch TensorFlow bots battle!</p>
                    <p className="text-gray-400 text-sm">Player tank vs AI-controlled enemies</p>
                    <Button color="primary" size="lg" onClick={handleStart} isLoading={isLoading}>
                        {isLoading ? 'Loading TensorFlow...' : 'Start Game'}
                    </Button>
                </Card>
            )}
        </div>
    );
}
