import { CSSProperties, useCallback, useEffect, useRef, useState } from 'react';
import { useObservable } from '../../../../../../lib/React/useSyncObservable.ts';
import { TestGameState$, incrementEnemyCount } from '../../State/GameState.ts';
import { startTestGame, spawnEnemy, exitTestGame } from '../../State/gameMethods.ts';
import { setRenderTarget } from '../../State/RenderTarget.ts';
import { Button } from '../../../Arena/UI/Components/Button.tsx';
import { Card } from '../../../Arena/UI/Components/Card.tsx';

export function BaseScreen({ className, style }: { className?: string, style?: CSSProperties }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isSpawning, setIsSpawning] = useState(false);
    const { isStarted, enemyCount } = useObservable(TestGameState$, TestGameState$.value);

    useEffect(() => {
        setRenderTarget(canvasRef.current);
        return () => setRenderTarget(null);
    }, [canvasRef.current]);

    useEffect(() => {
        if (!isStarted && canvasRef.current) {
            startTestGame();
        }
    }, [isStarted, canvasRef.current]);

    const handleStart = useCallback(async () => {
        setIsLoading(true);
        await startTestGame();
        setIsLoading(false);
    }, []);

    const handleStop = useCallback(() => {
        exitTestGame();
    }, []);

    const handleAddEnemy = useCallback(async () => {
        setIsSpawning(true);
        await spawnEnemy();
        incrementEnemyCount();
        setIsSpawning(false);
    }, []);

    return (
        <div className={`${className} flex items-center justify-center relative`} style={style}>
            <canvas
                ref={canvasRef}
                className="absolute w-full h-full"
            />

            {/* Controls when game is running */}
            {isStarted && (
                <div className="absolute top-4 left-4 flex gap-4 items-center">
                    <div className="bg-black/70 backdrop-blur-sm px-4 py-2 rounded-lg border border-emerald-500/30">
                        <span className="text-emerald-400 font-mono text-sm">Enemies: {enemyCount}</span>
                    </div>
                    <Button 
                        color="success" 
                        size="sm" 
                        onClick={handleAddEnemy}
                        isLoading={isSpawning}
                    >
                        + Add Enemy
                    </Button>
                    <Button color="danger" size="sm" onClick={handleStop}>
                        Exit
                    </Button>
                </div>
            )}

            {/* Start Screen */}
            {!isStarted && (
                <Card className="absolute p-8 flex flex-col items-center gap-6 bg-gradient-to-br from-slate-900 via-emerald-950 to-slate-900 border border-emerald-500/20">
                    <div className="flex flex-col items-center gap-2">
                        <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
                            Test Arena
                        </h1>
                        <p className="text-slate-400">Spawn and test AI enemies</p>
                    </div>
                    
                    <div className="flex flex-col items-center gap-2 text-sm text-slate-500">
                        <p>🎮 WASD - Move</p>
                        <p>🖱️ Mouse - Aim & Shoot</p>
                    </div>
                    
                    <Button 
                        color="success" 
                        size="lg" 
                        onClick={handleStart} 
                        isLoading={isLoading}
                        className="font-semibold"
                    >
                        {isLoading ? 'Loading...' : 'Start Test'}
                    </Button>
                </Card>
            )}
        </div>
    );
}

