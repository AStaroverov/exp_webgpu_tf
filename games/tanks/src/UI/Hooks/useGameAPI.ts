import { Game } from '../../Game/createGame.ts';
import { useEffect, useMemo, useState } from 'react';
import { createGameAPI } from '../../Game/createGameAPI.ts';
import { frameTasks } from '../../../../../lib/TasksScheduler/frameTasks.ts';

export type GameTankState = {
    id: number,
    health: number,
}
export type GameState = {
    tankEids: number[],
    tanks: GameTankState[],
}

export function useGameAPI(game: Game) {
    const [state, setState] = useState<GameState>({
        tankEids: [],
        tanks: [],
    });
    const api = useMemo(() => createGameAPI(game), [game]);

    useEffect(() => {
        const stop = frameTasks.addInterval(() => {
            const tankEids = api.getTankEids();
            const tanks = tankEids.map((tankId) => {
                return {
                    id: tankId,
                    health: 100,
                };
            });

            setState({ tankEids, tanks });
        }, 10);

        return stop;
    }, [api]);

    return { state, api };
}