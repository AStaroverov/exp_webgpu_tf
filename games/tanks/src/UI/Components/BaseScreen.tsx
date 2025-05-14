import mergeRefs from 'merge-refs';
import { useCallback, useEffect, useRef, useState } from 'react';
import { createGame } from '../../Game/createGame.ts';
import { useMeasure } from '@uidotdev/usehooks';
import { GarageScreen } from './GarageScreen.tsx';
import { frameTasks } from '../../../../../lib/TasksScheduler/frameTasks.ts';
import { max } from '../../../../../lib/math.ts';
import { useGameAPI } from '../Hooks/useGameAPI.ts';

const game = createGame({ width: 1000, height: 1000, withPlayer: false });

export function BaseScreen({ className, style }: { className?: string, style?: React.CSSProperties }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [measureRef, { width, height }] = useMeasure<HTMLCanvasElement>();

    const [gameState, setGameState] = useState({
        started: false,
    });

    useEffect(() => {
        let time = Date.now();
        const step = gameState.started ? 1 : 10;
        const stop = frameTasks.addInterval(() => {
            const t = Date.now();
            const delta = max(step * 16.6667, t - time);
            time = t;
            game?.gameTick(delta);
        }, step);

        return () => {
            stop();
        };
    }, [gameState.started]);

    useEffect(() => {
        game?.setRenderTarget(canvasRef.current);
    }, [canvasRef.current, canvasRef.current]);

    useEffect(() => {
        if (canvasRef.current && width && height) {
            canvasRef.current.style.width = width + 'px';
            canvasRef.current.style.height = height + 'px';
        }
    }, [width, height]);

    const { state, api } = useGameAPI(game);

    const handleAddTank = useCallback(() => {
        if (state.tankEids.length >= api.maxTanks || gameState.started) {
            return;
        }

        api.addTank(state.tankEids.length);
    }, [gameState, state.tankEids.length]);

    const handleStart = useCallback(() => {
        setGameState((s) => ({ ...s, started: true }));
        api.startGame();
    }, []);

    return (
        <div className={ `${ className } flex items-center justify-center` } style={ style }>
            <canvas
                ref={ mergeRefs(canvasRef, measureRef) }
                className={ `absolute size-full transition-all origin-left
                    ${ gameState.started ? 'left-0' : 'left-20 scale-50' }
                ` }
            />
            { game && <GarageScreen
                className={ `absolute transition-all ${ gameState.started ? 'left-0 opacity-0 ' : 'right-60' }` }
                tanks={ state.tanks }
                handleStart={ !gameState.started ? handleStart : undefined }
                handleAddTank={ state.tanks.length < api.maxTanks ? handleAddTank : undefined }
            /> }
        </div>
    );
}


