import { addTank as _addTank, getTankEids } from './engineMethods.ts';
import { toggleGame } from './GameState.ts';
import { destroyEngine } from './engine.ts';
import { activateBots, destroyBots, finalizeGameState } from './gameMethods.ts';

export const PLAYER_TEAM_ID = 0;

export const addTank = () => {
    const tankEids = getTankEids();
    _addTank(tankEids.length, PLAYER_TEAM_ID);
};

export const startGame = async () => {
    await finalizeGameState();
    activateBots();
    toggleGame(true);
};

export const exitGame = () => {
    destroyBots();
    destroyEngine();
    toggleGame(false);
};
